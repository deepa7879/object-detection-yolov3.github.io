# 🧠 Object Detection Microservice

A simple **Flask-based object detection microservice** powered by a lightweight **YOLOv3** model (CPU compatible).
It allows users to upload an image via a web interface and view detected objects with bounding boxes.
All results — annotated images and detection data — are saved inside the `results/` folder.

---

## 🚀 Features

* Object detection using **YOLOv3** (runs on CPU, no GPU required)
* **Flask web interface** for image uploads
* **Automatic output saving:**

  * 🖼️ Annotated image → `results/<uuid>.jpg`
  * 🧾 Detection results (JSON) → `results/<uuid>.json`
* **One-line Docker setup** for easy deployment

---

## 🧱 Project Structure

```
object-detection-microservices/
│
├── app.py                # Flask app with detection logic
├── requirements.txt       # Dependencies
├── Dockerfile             # Container setup
│
├── templates/
│   └── index.html         # Web upload UI
│
├── yolov3/                # Model files
│   ├── yolov3.cfg
│   ├── yolov3.weights
│   └── coco.names
│
└── results/               # Output folder (mounted locally)
│
└── README.md
```

---

## ⚙️ Model Setup

This project uses the **YOLOv3** model for object detection.

The model configuration (`yolov3.cfg`) and class labels (`coco.names`) are included in the repository under the `yolov3/` folder.

The large weights file (`yolov3.weights`, ~236 MB) **is not uploaded to GitHub** due to file size limits.

👉 When you run the application for the first time, it will **automatically download** the `yolov3.weights` file if not already present.

---

## 🐳 Running with Docker

### 1️⃣ Build the Docker Image

Run this command from your **project root folder**:

```bash
docker build -t object-detection-microservices .
```

---



### 2️⃣ Run the Container

#### 🪟 **Windows PowerShell**

```bash
docker run -p 5000:5000 -v "PATH:/app/results" object-detection-microservices
```

> ⚠️ Replace `PATH (for example: "F:/code/object-detection-microservices/results:/app/results")` with your actual path if it’s different.
> Use **forward slashes (`/`)** in paths when possible to avoid Docker path errors.

#### 🐧 **Ubuntu / Linux**

```bash
docker run -p 5000:5000 -v $(pwd)/results:/app/results object-detection-microservices
```

#### 🍎 **macOS**

```bash
docker run -p 5000:5000 -v $(pwd)/results:/app/results object-detection-microservices
```

---

### 3️⃣ Access the Application

Once the container starts successfully, open your browser at:

👉 [http://localhost:5000](http://localhost:5000)

Upload an image and wait for detection to complete.

---

## 📁 Output Files

After running detection, your **local `results/` folder** will contain:

```
results/
├── <uuid>.jpg   ← Image with bounding boxes
└── <uuid>.json  ← Detection results (object names, confidence, coordinates)
```

---

## 🧾 Notes

* Designed to run entirely on **CPU**.
* Make sure model files (`yolov3.cfg`, `yolov3.weights`, `coco.names`) are present inside the `yolov3/` folder.
* Use Docker for consistent environment replication on any machine.

---

**Author:**  Deepa
