# server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io, json, numpy as np, os, sys

MODEL_PATH = os.getenv("MODEL_PATH", "model.tflite")
LABELS_PATH = os.getenv("LABELS_PATH", "labels.txt")
NUTRIENTS_PATH = os.getenv("NUTRIENTS_PATH", "nutrients.json")
NUTRIENTS_TEMPLATE_PATH = os.getenv("NUTRIENTS_TEMPLATE_PATH", "nutrients_template.json")
PORT = int(os.getenv("PORT", 5000))

app = Flask(__name__)
CORS(app)

# --- Load required labels.txt (must be provided by you)
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        LABELS = [l.strip() for l in f.readlines() if l.strip()]
    if len(LABELS) == 0:
        print("labels.txt found but empty. Please populate with one label per line.")
        LABELS = []
else:
    LABELS = []

# --- Load nutrients.json if present; otherwise create a template file with default fields
NUTRIENTS = {}
if os.path.exists(NUTRIENTS_PATH):
    try:
        with open(NUTRIENTS_PATH, "r", encoding="utf-8") as f:
            NUTRIENTS = json.load(f)
    except Exception as e:
        print("Failed to load nutrients.json:", str(e))
        NUTRIENTS = {}
else:
    # create a template mapping only if labels are available
    if LABELS:
        template = {}
        for lbl in LABELS:
            template[lbl] = {
                "serving_g": None,
                "per_serving": {"kcal": None, "protein_g": None, "fat_g": None, "carbs_g": None, "fiber_g": None},
                "per_100g": {"kcal": None, "protein_g": None, "fat_g": None, "carbs_g": None, "fiber_g": None},
                "notes": "Fill values or replace with verified nutrients.json"
            }
        try:
            with open(NUTRIENTS_TEMPLATE_PATH, "w", encoding="utf-8") as f:
                json.dump(template, f, indent=2, ensure_ascii=False)
            print(f"Wrote nutrients template to {NUTRIENTS_TEMPLATE_PATH}. Fill values and save as {NUTRIENTS_PATH} to enable nutrient reporting.")
        except Exception as e:
            print("Failed to write nutrients template:", str(e))
    else:
        print("No labels.txt present; nutrients template not created.")

# --- TFLite interpreter setup (real inference if model.tflite is present)
INTERPRETER_AVAILABLE = False
interpreter = None
input_det = None
output_det = None

if os.path.exists(MODEL_PATH):
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        interpreter = Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_det = interpreter.get_input_details()[0]
        output_det = interpreter.get_output_details()[0]
        INTERPRETER_AVAILABLE = True
        print("Loaded TFLite model:", MODEL_PATH)
    except Exception as e:
        print("Failed to load TFLite interpreter or model:", str(e))
        INTERPRETER_AVAILABLE = False
else:
    print("No model.tflite found; server will refuse inference until a model is added.")

# --- Helpers
def preprocess_pil(img):
    if INTERPRETER_AVAILABLE and input_det is not None:
        h, w = int(input_det['shape'][1]), int(input_det['shape'][2])
    else:
        h, w = 224, 224
    img = img.resize((w, h)).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, 0)
    if INTERPRETER_AVAILABLE and input_det is not None and input_det['dtype'] == np.uint8:
        q = input_det.get('quantization', (1.0, 0))
        scale, zp = q if len(q) == 2 else (1.0, 0)
        arr = (arr / scale + zp).astype(np.uint8)
    return arr

def run_inference(img_bytes):
    if not INTERPRETER_AVAILABLE or interpreter is None:
        return None
    img = Image.open(io.BytesIO(img_bytes))
    inp = preprocess_pil(img)
    interpreter.set_tensor(input_det['index'], inp)
    interpreter.invoke()
    out = interpreter.get_tensor(output_det['index']).squeeze()
    if out.ndim == 1 and out.size > 1:
        idx = int(np.argmax(out))
        conf = float(np.max(out))
    else:
        idx = 0
        conf = float(out.item()) if hasattr(out, 'item') else 0.0
    return idx, float(conf)

def scale_nutrients(entry, multiplier=1.0, grams=None):
    if grams is not None and grams > 0 and entry.get("per_100g") and entry["per_100g"].get("kcal") is not None:
        base = entry["per_100g"]
        factor = grams / 100.0
    else:
        base = entry.get("per_serving") or entry
        factor = multiplier
    return {
        "kcal": int(round((base.get("kcal") or 0) * factor)),
        "protein_g": round((base.get("protein_g") or 0.0) * factor, 2),
        "fat_g": round((base.get("fat_g") or 0.0) * factor, 2),
        "carbs_g": round((base.get("carbs_g") or 0.0) * factor, 2),
        "fiber_g": round((base.get("fiber_g") or 0.0) * factor, 2)
    }

# --- Minimal HTML UI
INDEX_HTML = """<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Flask Food Inference</title>
    <style>
      body {{ font-family: Arial, sans-serif; margin: 24px; }}
      .box {{ max-width:720px; }}
      pre {{ background:#f4f4f4; padding:12px; border-radius:6px; overflow:auto; }}
      label {{ display:block; margin-top:8px; }}
    </style>
  </head>
  <body>
    <div class="box">
      <h2>Flask Food Inference</h2>
      <p>Labels loaded: {labels_count}. Model mode: {model_mode}.</p>
      <form method="post" action="/predict" enctype="multipart/form-data">
        <label>Select image to upload:</label>
        <input type="file" name="file" accept="image/*" required>
        <label>Portion multiplier (e.g., 1.0 = one serving):</label>
        <input type="number" step="0.1" min="0.1" name="mult" value="1.0">
        <label>Or enter grams (overrides multiplier):</label>
        <input type="number" step="1" min="1" name="grams" placeholder="grams">
        <br><br>
        <button type="submit">Upload & Predict</button>
      </form>
      <hr>
      <div id="result">{result_block}</div>
      <hr>
      <p>Provide labels.txt (one label per line) at the project root. Provide nutrients.json to enable nutrient reporting. A template may have been created at {template_path}.</p>
    </div>
  </body>
</html>"""

def render_index(result_json=None):
    if result_json is None:
        block = "<p>No prediction yet.</p>"
    else:
        pretty = json.dumps(result_json, indent=2)
        pretty = pretty.replace("<", "&lt;").replace(">", "&gt;")
        block = f"<h3>Prediction result</h3><pre>{pretty}</pre>"
    html = INDEX_HTML.replace("{result_block}", block)
    html = html.replace("{labels_count}", str(len(LABELS)))
    html = html.replace("{model_mode}", "real" if INTERPRETER_AVAILABLE else "no-model")
    html = html.replace("{template_path}", NUTRIENTS_TEMPLATE_PATH)
    return html

# --- Routes
@app.route("/", methods=["GET"])
def index():
    if not LABELS:
        msg = "<p><strong>labels.txt is missing or empty.</strong> Place a labels.txt file (one label per line) in the project root and reload.</p>"
        return INDEX_HTML.replace("{result_block}", msg).replace("{labels_count}", "0").replace("{model_mode}", "no-model").replace("{template_path}", NUTRIENTS_TEMPLATE_PATH), 200
    return render_index(), 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_present": os.path.exists(MODEL_PATH),
        "labels_count": len(LABELS),
        "nutrients_loaded": bool(NUTRIENTS),
        "note": "Provide labels.txt and nutrients.json for full functionality"
    })

@app.route("/predict", methods=["POST"])
def predict():
    if not LABELS:
        return jsonify({"detail": "labels.txt missing or empty; put labels.txt in project root"}), 400
    if not os.path.exists(MODEL_PATH):
        return jsonify({"detail": "no model found; place model.tflite in project root"}), 400
    if 'file' not in request.files:
        return jsonify({"detail": "file missing"}), 400

    f = request.files['file']
    try:
        img_bytes = f.read()
        Image.open(io.BytesIO(img_bytes))
    except Exception:
        return jsonify({"detail": "invalid image"}), 400

    # run model
    try:
        res = run_inference(img_bytes)
        if res is None:
            raise RuntimeError("interpreter not ready")
        idx, conf = res
        if idx < 0 or idx >= len(LABELS):
            label = f"class_{idx}"
        else:
            label = LABELS[idx]
    except Exception as e:
        return jsonify({"detail": "inference error", "error": str(e)}), 500

    # parse scaling inputs
    grams = None
    mult = 1.0
    try:
        if request.form.get("grams"):
            grams = float(request.form.get("grams"))
        elif request.form.get("mult"):
            mult = float(request.form.get("mult"))
    except Exception:
        grams = None
        mult = 1.0

    entry = NUTRIENTS.get(label)
    if entry is None:
        # nutrients.json not provided or label missing
        return jsonify({
            "label": label,
            "confidence": round(conf, 4),
            "detail": "nutrients not available for this label; provide nutrients.json or edit nutrients_template.json"
        }), 200

    scaled = scale_nutrients(entry, multiplier=mult, grams=grams)
    result = {
        "label": label,
        "confidence": round(conf, 4),
        "serving_or_scale": {"grams": grams, "multiplier": mult},
        "kcal": scaled["kcal"],
        "protein_g": scaled["protein_g"],
        "fat_g": scaled["fat_g"],
        "carbs_g": scaled["carbs_g"],
        "fiber_g": scaled["fiber_g"]
    }

    accept = request.headers.get("Accept", "")
    if "application/json" in accept:
        return jsonify(result)
    else:
        return render_index(result), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)
