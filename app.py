from flask import Flask, render_template, request, send_file, url_for
import os
import cv2
import numpy as np
import uuid
from PIL import Image
from io import BytesIO
from werkzeug.utils import secure_filename
import json
import urllib.request

# === CONFIG ===
MODEL_DIR = os.path.join(os.path.dirname(__file__), "yolov3")
CFG_PATH = os.path.join(MODEL_DIR, "yolov3.cfg")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "yolov3.weights")
NAMES_PATH = os.path.join(MODEL_DIR, "coco.names")

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Download YOLOv3 weights automatically if not found
if not os.path.exists(WEIGHTS_PATH):
    print("⬇️ Downloading YOLOv3 weights...")
    url = "https://data.pjreddie.com/files/yolov3.weights"
    urllib.request.urlretrieve(url, WEIGHTS_PATH)
    print("✅ Download complete!")

# Check all files exist before proceeding
for file_path in [WEIGHTS_PATH, CFG_PATH, NAMES_PATH]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Missing file: {file_path}. Please ensure it's in the 'yolov3/' folder.")

app = Flask(__name__)

# === LOAD MODEL ===
if not (os.path.exists(CFG_PATH) and os.path.exists(WEIGHTS_PATH)):
    raise FileNotFoundError("⚠️ YOLO files missing! Place yolov3.cfg and yolov3.weights in 'yolov3/' folder.")

with open(NAMES_PATH, "r") as f:
    CLASS_NAMES = [c.strip() for c in f.readlines()]

net = cv2.dnn.readNetFromDarknet(CFG_PATH, WEIGHTS_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layer_names = net.getLayerNames()
try:
    ln = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
except:
    ln = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


# === DETECTION FUNCTION ===
def detect_image(image_bytes, conf_thresh=0.5, nms_thresh=0.4):
    img = Image.open(BytesIO(image_bytes)).convert('RGB')
    np_img = np.array(img)[:, :, ::-1]  # RGB → BGR
    (H, W) = np_img.shape[:2]

    blob = cv2.dnn.blobFromImage(np_img, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxes, confidences, classIDs = [], [], []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > conf_thresh:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)
    results = []

    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            cls = CLASS_NAMES[classIDs[i]]
            results.append({
                "class": cls,
                "confidence": float(confidences[i]),
                "bbox": {"x": x, "y": y, "width": w, "height": h}
            })

    annotated = np_img.copy()
    for r in results:
        x, y, w, h = r['bbox'].values()
        label = f"{r['class']}:{r['confidence']:.2f}"
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(annotated, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    out_id = str(uuid.uuid4())
    out_image_path = os.path.join(RESULTS_DIR, out_id + '.jpg')
    out_json_path = os.path.join(RESULTS_DIR, out_id + '.json')

    cv2.imwrite(out_image_path, annotated)

    out_payload = {"id": out_id, "detections": results}
    with open(out_json_path, "w") as f:
        json.dump(out_payload, f, indent=2)

    return out_payload, out_image_path


# === ROUTES ===
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return 'no image', 400
    f = request.files['image']
    fname = secure_filename(f.filename)
    in_path = os.path.join(UPLOAD_DIR, fname)
    f.save(in_path)

    with open(in_path, 'rb') as img_file:
        img_bytes = img_file.read()

    data, out_image_path = detect_image(img_bytes)

    return render_template(
        'index.html',
        detections=data.get('detections', []),
        image_url=url_for('get_result', fname=os.path.basename(out_image_path)),
        json=data
    )


@app.route('/results/<fname>')
def get_result(fname):
    path = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(path):
        return 'not found', 404
    return send_file(path, mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
