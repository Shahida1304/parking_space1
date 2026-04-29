from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import os
import time
import threading
import cv2

app = Flask(__name__)

# ===============================
# FOLDERS
# ===============================
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# ===============================
# LOAD MODEL (only once)
# ===============================
model = YOLO("best1.pt")

# detection results storage
detection_stats = {"free": 0, "occupied": 0, "total": 0}


# ===============================
# GET LATEST IMAGE FROM FOLDER
# ===============================
def get_latest_image():
    files = [
        os.path.join(UPLOAD_FOLDER, f)
        for f in os.listdir(UPLOAD_FOLDER)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not files:
        return None

    # newest file by modification time
    latest_file = max(files, key=os.path.getmtime)
    return latest_file


# ===============================
# PROCESS IMAGE (MAIN DETECTION)
# ===============================
def process_image(image_path):
    global detection_stats

    try:
        print(f"[PROCESSING] {image_path}")

        results = model.predict(source=image_path, conf=0.4, save=False)

        # read original image
        img = cv2.imread(image_path)

        free = 0
        occupied = 0

        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        for box, cls in zip(boxes, classes):
            x1, y1, x2, y2 = map(int, box)

            # CLASS 0 = EMPTY → GREEN
            if int(cls) == 0:
                color = (0, 255, 0)
                free += 1
            else:
                color = (0, 0, 255)
                occupied += 1

            # draw ONLY rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        total = free + occupied

        detection_stats = {
            "free": free,
            "occupied": occupied,
            "total": total,
        }

        # save output image
        output_path = os.path.join(STATIC_FOLDER, "output.jpg")
        cv2.imwrite(output_path, img)

        print(f"[DONE] Free={free} Occupied={occupied} Total={total}")

    except Exception as e:
        print("Processing Error:", e)


# ===============================
# AUTO DETECTION THREAD
# ===============================
def auto_detect():
    last_processed = None

    while True:
        latest = get_latest_image()

        if latest and latest != last_processed:
            process_image(latest)
            last_processed = latest

        time.sleep(120)  # every 2 minutes


threading.Thread(target=auto_detect, daemon=True).start()


# ===============================
# MAIN PAGE
# ===============================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")

        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # immediate detection after upload
            process_image(filepath)

    return render_template(
        "index.html",
        free=detection_stats["free"],
        occupied=detection_stats["occupied"],
        total=detection_stats["total"],
    )


# ===============================
# LIVE STATS API
# ===============================
@app.route("/stats")
def stats():
    return jsonify(detection_stats)


# ===============================
if __name__ == "__main__":
    app.run(debug=True)
