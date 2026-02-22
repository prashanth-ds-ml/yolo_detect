import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO

# -------------------------
# Load model once (global)
# -------------------------
MODEL_NAME = "yolov8n.pt"
model = YOLO(MODEL_NAME)

CLASS_OF_INTEREST = "person"

def is_in_danger_zone(box, zone):
    """
    box: (x1, y1, x2, y2)
    zone: ((zx1, zy1), (zx2, zy2))
    overlap logic: any partial overlap triggers True
    """
    x1, y1, x2, y2 = box
    (zx1, zy1), (zx2, zy2) = zone

    overlap_x = (x1 < zx2) and (x2 > zx1)
    overlap_y = (y1 < zy2) and (y2 > zy1)
    return overlap_x and overlap_y

def process_frame(frame, zx1, zy1, zx2, zy2, conf_thres):
    """
    frame: numpy array (H, W, 3) from Gradio webcam (RGB)
    returns: annotated frame (RGB), status markdown
    """
    if frame is None:
        return None, "Waiting for webcam input…"

    # Gradio gives RGB; OpenCV drawing expects BGR
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]

    # Clamp and fix zone coordinates
    zx1 = int(np.clip(zx1, 0, w - 1))
    zx2 = int(np.clip(zx2, 0, w - 1))
    zy1 = int(np.clip(zy1, 0, h - 1))
    zy2 = int(np.clip(zy2, 0, h - 1))

    if zx2 < zx1:
        zx1, zx2 = zx2, zx1
    if zy2 < zy1:
        zy1, zy2 = zy2, zy1

    danger_zone = ((zx1, zy1), (zx2, zy2))

    # Draw danger zone
    cv2.rectangle(bgr, danger_zone[0], danger_zone[1], (0, 0, 255), 2)

    # Run YOLO (on original RGB frame or BGR — ultralytics handles numpy arrays)
    results = model.predict(source=frame, conf=float(conf_thres), verbose=False)

    alert_triggered = False
    persons_in_zone = 0
    persons_total = 0

    for r in results:
        names = r.names
        if r.boxes is None:
            continue

        boxes_xyxy = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes.xyxy, "cpu") else np.array(r.boxes.xyxy)
        cls_ids = r.boxes.cls.cpu().numpy() if hasattr(r.boxes.cls, "cpu") else np.array(r.boxes.cls)
        confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes.conf, "cpu") else np.array(r.boxes.conf)

        for (x1, y1, x2, y2), cls_id, cf in zip(boxes_xyxy, cls_ids, confs):
            class_name = names[int(cls_id)]
            if class_name != CLASS_OF_INTEREST:
                continue

            persons_total += 1

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # Draw person bbox
            cv2.rectangle(bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"{class_name}: {float(cf):.2f}"
            cv2.putText(bgr, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Zone overlap check
            if is_in_danger_zone((x1, y1, x2, y2), danger_zone):
                alert_triggered = True
                persons_in_zone += 1

    if alert_triggered:
        cv2.putText(
            bgr,
            f"ALERT! {persons_in_zone} person(s) in danger zone",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3
        )
        status = f"## 🔴 ALERT\n**{persons_in_zone}** person(s) inside danger zone.\n\nTotal persons detected: **{persons_total}**"
    else:
        status = f"## ✅ SAFE\nNo person inside danger zone.\n\nTotal persons detected: **{persons_total}**"

    # Convert back to RGB for Gradio output
    out_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return out_rgb, status


with gr.Blocks(title="YOLOv8 Danger Zone (Webcam)") as demo:
    gr.Markdown(
        "# YOLOv8 Danger Zone Detection (Webcam)\n"
        "Use your webcam, define a rectangular danger zone, and detect if any **person** enters it.\n\n"
        "**Note:** On Hugging Face Spaces, server-side audio (pygame) isn’t reliable. We show a clear on-screen alert instead."
    )

    with gr.Row():
        with gr.Column():
            cam = gr.Image(
                label="Webcam Input",
                sources=["webcam"],
                type="numpy"
            )
        with gr.Column():
            out = gr.Image(label="Annotated Output", type="numpy")
            status_md = gr.Markdown("Waiting for webcam input…")

    with gr.Accordion("Danger Zone Controls", open=True):
        with gr.Row():
            zx1 = gr.Slider(0, 1280, value=100, step=1, label="Zone X1 (left)")
            zy1 = gr.Slider(0, 720, value=100, step=1, label="Zone Y1 (top)")
        with gr.Row():
            zx2 = gr.Slider(0, 1280, value=400, step=1, label="Zone X2 (right)")
            zy2 = gr.Slider(0, 720, value=400, step=1, label="Zone Y2 (bottom)")
        conf = gr.Slider(0.1, 0.9, value=0.35, step=0.05, label="Confidence Threshold")

    # Stream webcam frames to backend (Gradio 5 streaming)
    cam.stream(
        fn=process_frame,
        inputs=[cam, zx1, zy1, zx2, zy2, conf],
        outputs=[out, status_md],
        stream_every=0.1  # approx 10 fps snapshots (depends on device/network)
    )

demo.queue().launch()