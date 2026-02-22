
---

# 🚨 YOLOv8 Danger Zone Detection (Port Safety Demo)

## 📌 Overview

This project demonstrates a **real-time computer vision safety system** that detects when a **person enters a predefined danger zone** and raises an alert.

It is designed as a **proof-of-concept (PoC)** for **ports, harbors, industrial sites, and construction zones**, where restricted areas must be actively monitored to prevent accidents.

Hugging Face demo: https://huggingface.co/spaces/KPrashanth/Yolo_detect
---

## 🎯 Problem Statement

Ports and industrial environments contain **high-risk zones** such as:

* Crane operating areas
* Container movement lanes
* Dock edges and restricted zones
* Heavy machinery pathways

Manual monitoring is:

* ❌ Error-prone
* ❌ Not scalable
* ❌ Delayed in response

---

## 💡 Solution

This system uses **YOLOv8 (Ultralytics)** for real-time object detection and:

* Identifies **people** in the video stream
* Checks if they **enter a defined danger zone**
* Triggers an **instant visual alert**

---

## 🚀 Features

### ✅ Real-Time Detection

* Detects **persons using YOLOv8**
* Works on **live webcam (browser-based via Gradio)**

### ✅ Configurable Danger Zone

* Define zone using sliders:

  * `(X1, Y1)` → Top-left
  * `(X2, Y2)` → Bottom-right

### ✅ Smart Overlap Detection

* Triggers alert if **any part of person enters zone**
* Uses **bounding box intersection logic**

### ✅ Visual Alert System

* Red alert banner on screen:

  ```
  ALERT! Person in danger zone
  ```
* Shows:

  * Number of persons detected
  * Number inside zone

### ✅ Lightweight & Deployable

* Runs on **CPU (YOLOv8n)**
* Easily deployable on **Hugging Face Spaces**

---

## 🧠 How It Works

### Step 1: Input Stream

* Webcam input captured via **Gradio**

### Step 2: Object Detection

* YOLOv8 detects objects
* Filters only:

  ```
  class = "person"
  ```

### Step 3: Danger Zone Logic

Each detected person:

* Bounding box → `(x1, y1, x2, y2)`
* Check overlap with danger zone:

```python
overlap_x = (x1 < zx2) and (x2 > zx1)
overlap_y = (y1 < zy2) and (y2 > zy1)
```

If both are True → 🚨 ALERT

### Step 4: Output

* Annotated video frame
* Status message:

  * ✅ SAFE
  * 🔴 ALERT

---

## 🖥️ Demo Interface

* 🎥 Live webcam feed
* 📦 Bounding boxes on detected persons
* 🔴 Red danger zone overlay
* ⚙️ Sliders to adjust zone dynamically
* 📊 Real-time alert status

---

## 📦 Project Structure

```
.
├── app.py              # Main Gradio application
├── requirements.txt    # Dependencies
├── README.md           # Documentation
```

---

## ⚙️ Installation (Local)

```bash
git clone <your-repo-url>
cd <repo>

pip install -r requirements.txt
python app.py
```

---

## 🌐 Deployment (Hugging Face Spaces)

### 1. Create Space

* Select:

  * SDK: **Gradio**

### 2. Upload Files

* `app.py`
* `requirements.txt`
* `README.md`

### 3. Add to README (top block)

```yaml
---
title: YOLO Danger Zone Detection
emoji: 🚨
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 5.23.1
app_file: app.py
---
```

---

## 📊 Use Cases

### 🚢 Ports & Harbors

* Restricted dock zones
* Crane operating areas
* Container loading lanes

### 🏗️ Construction Sites

* Hazard zones near machinery
* No-entry areas

### 🏭 Industrial Safety

* Factory floor monitoring
* Worker safety compliance

### 🚧 Traffic & Surveillance

* Pedestrian intrusion detection
* Restricted lane monitoring

---

## 🔮 Future Improvements

* 🔊 Audio alert (client-side browser sound)
* 📩 SMS / Email / WhatsApp alerts
* 🎯 Multi-class detection (helmet, vehicle, forklift)
* 🎥 CCTV / RTSP stream integration
* 🧠 Multi-zone support per camera
* 📈 Dashboard for alerts & analytics
* ⚡ GPU optimization for multi-camera setup

---

## ⚠️ Limitations

* Depends on camera quality & lighting
* May produce false positives in crowded scenes
* Single-class detection (person only)
* No persistence tracking (yet)

---

## 🧩 Tech Stack

* **YOLOv8 (Ultralytics)** – Object Detection
* **OpenCV** – Image Processing
* **Gradio** – UI & Deployment
* **NumPy** – Data handling

---

## 👨‍💻 Author

**Katakam Prashanth**

* AI / ML Engineer 
* Focus: Computer Vision, LLMs, Real-world AI Systems

---

## ⭐ Key Takeaway

> This demo proves that **AI-powered vision systems can proactively monitor restricted zones and prevent accidents in real-time**, making industrial environments significantly safer.

---
