import cv2
import mediapipe as mp
import numpy as np
from fatigue_detector import get_ear_from_landmarks
from seatbelt_detector import is_seatbelt_on
import time
import csv
import winsound

mp_face_mesh = mp.solutions.face_mesh

EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 20

# For logging
LOG_FILE = "drowsiness_log.csv"

def log_event(event):
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), event])

def main():
    cap = cv2.VideoCapture(0)
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
    drowsy_counter = 0
    seatbelt_on = is_seatbelt_on()
    alert_played = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                landmarks = [np.array([lm.x * w, lm.y * h]) for lm in face_landmarks.landmark]
                ear = get_ear_from_landmarks(face_landmarks.landmark)

                # Draw eye landmarks
                for idx in [33, 160, 158, 133, 153, 144, 263, 387, 385, 362, 380, 373]:
                    x = int(face_landmarks.landmark[idx].x * w)
                    y = int(face_landmarks.landmark[idx].y * h)
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                if ear < EAR_THRESHOLD:
                    drowsy_counter += 1
                else:
                    drowsy_counter = 0
                    alert_played = False

                if drowsy_counter >= CONSEC_FRAMES:
                    cv2.putText(frame, "DROWSINESS ALERT!", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
                    if not alert_played:
                        try:
                            # winsound.Beep(frequency, duration_ms)
                            winsound.Beep(1500, 500)
                        except Exception:
                            pass
                        log_event("DROWSINESS ALERT")
                        alert_played = True
                cv2.putText(frame, f"EAR: {ear:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        else:
            drowsy_counter = 0
            alert_played = False

        # Seatbelt simulation
        if not seatbelt_on:
            cv2.putText(frame, "SEATBELT NOT DETECTED!", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)

        cv2.imshow('Driver Monitor', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            seatbelt_on = not seatbelt_on  # Toggle seatbelt status

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
