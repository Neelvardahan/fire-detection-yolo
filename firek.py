import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np
import threading
import pygame
import time

# Load models
detection_model = YOLO('best.pt')
classification_model = YOLO('bestnew.pt')

# Initialize alarm
pygame.mixer.init()
pygame.mixer.music.load("fire-alarm-2.mp3")

# Track alarm and consecutive fire frames
consecutive_fire_frames = 0
fire_threshold = 3  # Trigger alarm after 3 consecutive fire frames
alarm_cooldown = 5  # seconds between alarms
last_alarm_time = 0

# Alarm sound trigger
def play_alarm():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play()

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, frame= cap.read()
    if not ret:
        break 

    detections = detection_model(frame)
    fire_in_this_frame = False

    for det in detections:
        for box in det.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls == 0 and conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                crop_resized = cv2.resize(crop, (224, 224))
                crop_pil = Image.fromarray(cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB))

                # Classify
                class_result = classification_model(crop_pil)
                label = class_result[0].names[class_result[0].probs.top1]
                confidence = class_result[0].probs.top1conf

                # Color and alarm logic
                if label.lower() == "fire":
                    color = (0, 0, 255)  # Red
                    fire_in_this_frame = True
                elif label.lower() == "smoke":
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 255, 0)  # Green

                # Draw
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{label} ({confidence:.2f})', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Handle consecutive fire frame tracking
    if fire_in_this_frame:
        consecutive_fire_frames += 1
    else:
        consecutive_fire_frames = 0

    # Alarm trigger
    if consecutive_fire_frames >= fire_threshold:
        current_time = time.time()
        if current_time - last_alarm_time > alarm_cooldown:
            threading.Thread(target=play_alarm, daemon=True).start()
            last_alarm_time = current_time
            consecutive_fire_frames = 0  # Reset after triggering

    # Display
    cv2.imshow("🔥 Hybrid Fire Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
