
import cv2
import mediapipe as mp
import numpy as np
from fatigue_detector import get_ear_from_landmarks
import time
import csv
import winsound
import math

mp_face_mesh = mp.solutions.face_mesh


EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 20
MAR_THRESHOLD = 0.7  # Mouth aspect ratio threshold for yawn detection
YAWN_CONSEC_FRAMES = 15
BLINK_CONSEC_FRAMES = 3
BLINKS_PER_MIN_THRESHOLD = 20  # Blinks per minute threshold for fatigue
HEAD_PITCH_THRESHOLD = 30  # degrees, for nodding detection (less sensitive)

# For logging
LOG_FILE = "drowsiness_log.csv"

def log_event(event):
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), event])


def mouth_aspect_ratio(landmarks, w, h):
    # Improved MAR using 6 vertical and 2 horizontal points (outer mouth)
    # Vertical pairs: (13, 14), (312, 82), (311, 87)
    # Horizontal: (61, 291)
    p13 = np.array([landmarks[13].x * w, landmarks[13].y * h])
    p14 = np.array([landmarks[14].x * w, landmarks[14].y * h])
    p312 = np.array([landmarks[312].x * w, landmarks[312].y * h])
    p82 = np.array([landmarks[82].x * w, landmarks[82].y * h])
    p311 = np.array([landmarks[311].x * w, landmarks[311].y * h])
    p87 = np.array([landmarks[87].x * w, landmarks[87].y * h])
    left = np.array([landmarks[61].x * w, landmarks[61].y * h])
    right = np.array([landmarks[291].x * w, landmarks[291].y * h])
    # Average the three vertical distances
    vert1 = np.linalg.norm(p13 - p14)
    vert2 = np.linalg.norm(p312 - p82)
    vert3 = np.linalg.norm(p311 - p87)
    mar = (vert1 + vert2 + vert3) / 3.0 / np.linalg.norm(left - right)
    return mar

def get_head_pitch(landmarks, w, h):
    # Use nose tip (1), chin (152), left eye (33), right eye (263)
    nose = np.array([landmarks[1].x * w, landmarks[1].y * h])
    chin = np.array([landmarks[152].x * w, landmarks[152].y * h])
    dx = chin[0] - nose[0]
    dy = chin[1] - nose[1]
    angle = math.degrees(math.atan2(dy, dx))
    return angle

def main():
    cap = cv2.VideoCapture(0)
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
    drowsy_counter = 0
    yawn_counter = 0
    blink_counter = 0
    blink_start = None
    blinks = 0
    alert_played = False
    alert_cooldown = 0
    ALERT_COOLDOWN_FRAMES = 60  # Show alert for 2 seconds after eyes open (assuming ~30fps)
    last_blink_time = time.time()
    blink_history = []
    frame_count = 0
    start_time = time.time()
    neutral_head_pitch = None
    calibration_frames = 30  # Number of frames to calibrate neutral head position
    calibration_pitches = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        h, w, _ = frame.shape
        frame_count += 1

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                ear = get_ear_from_landmarks(landmarks)
                mar = mouth_aspect_ratio(landmarks, w, h)
                head_pitch = get_head_pitch(landmarks, w, h)

                # Calibrate neutral head pitch in the first few frames
                if neutral_head_pitch is None and len(calibration_pitches) < calibration_frames:
                    calibration_pitches.append(head_pitch)
                    if len(calibration_pitches) == calibration_frames:
                        neutral_head_pitch = sum(calibration_pitches) / len(calibration_pitches)
                    # Show calibration message
                    cv2.putText(frame, "Calibrating head position...", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
                # Draw eye and mouth landmarks
                for idx in [33, 160, 158, 133, 153, 144, 263, 387, 385, 362, 380, 373, 61, 291, 13, 14]:
                    x = int(landmarks[idx].x * w)
                    y = int(landmarks[idx].y * h)
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                # EAR for drowsiness
                if ear < EAR_THRESHOLD:
                    drowsy_counter += 1
                else:
                    if drowsy_counter >= CONSEC_FRAMES:
                        alert_cooldown = ALERT_COOLDOWN_FRAMES
                    drowsy_counter = 0
                    # Don't reset alert_played immediately; let cooldown handle it

                # MAR for yawn
                if mar > MAR_THRESHOLD:
                    yawn_counter += 1
                else:
                    yawn_counter = 0

                # Head pose (pitch) - use deviation from neutral
                nodding = False
                pitch_deviation = 0
                if neutral_head_pitch is not None:
                    pitch_deviation = abs(head_pitch - neutral_head_pitch)
                    nodding = pitch_deviation > HEAD_PITCH_THRESHOLD

                # Blink detection (EAR below threshold for a few frames)
                if ear < EAR_THRESHOLD:
                    blink_counter += 1
                else:
                    if blink_counter >= BLINK_CONSEC_FRAMES:
                        blinks += 1
                        blink_history.append(time.time())
                    blink_counter = 0

                # Blinks per minute
                current_time = time.time()
                blink_history = [t for t in blink_history if current_time - t < 60]
                blinks_per_min = len(blink_history)

                # Drowsiness score calculation
                # Weights: eye closure (0.4), yawn (0.2), nodding (0.2), blink rate (0.2)
                score = 0.0
                # Eye closure: normalized by how long eyes are closed
                score += min(drowsy_counter / CONSEC_FRAMES, 1.0) * 0.4
                # Yawn: normalized by how long mouth is open
                score += min(yawn_counter / YAWN_CONSEC_FRAMES, 1.0) * 0.2
                # Nodding: binary
                if neutral_head_pitch is not None and nodding:
                    score += 0.2
                # Blink rate: normalized by threshold
                score += min(blinks_per_min / BLINKS_PER_MIN_THRESHOLD, 1.0) * 0.2

                # Fatigue alerts
                fatigue_alert = False
                alert_msgs = []
                if score >= 0.7:
                    alert_msgs.append("DROWSINESS ALERT!")
                    fatigue_alert = True
                if drowsy_counter >= CONSEC_FRAMES:
                    alert_msgs.append("EYES CLOSED!")
                if yawn_counter >= YAWN_CONSEC_FRAMES:
                    alert_msgs.append("YAWNING DETECTED!")
                if neutral_head_pitch is not None and nodding:
                    alert_msgs.append("HEAD NODDING!")
                if blinks_per_min > BLINKS_PER_MIN_THRESHOLD:
                    alert_msgs.append("EXCESSIVE BLINKING!")

                y_offset = 60
                for msg in alert_msgs:
                    cv2.putText(frame, msg, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                    y_offset += 40

                if fatigue_alert and (not alert_played or alert_cooldown > 0):
                    try:
                        winsound.Beep(1500, 500)
                    except Exception:
                        pass
                    log_event("FATIGUE ALERT: " + ", ".join(alert_msgs))
                    alert_played = True
                    alert_cooldown = ALERT_COOLDOWN_FRAMES

                # Decrement cooldown and reset alert_played if needed
                if alert_cooldown > 0:
                    alert_cooldown -= 1
                else:
                    alert_played = False

                # Display metrics
                cv2.putText(frame, f"EAR: {ear:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(frame, f"MAR: {mar:.2f}", (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(frame, f"Blinks/min: {blinks_per_min}", (30, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                if neutral_head_pitch is not None:
                    cv2.putText(frame, f"Head Pitch: {head_pitch:.1f} (Δ{pitch_deviation:.1f})", (200, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                else:
                    cv2.putText(frame, f"Head Pitch: {head_pitch:.1f} (calibrating)", (200, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                # Show drowsiness score
                cv2.putText(frame, f"Drowsiness Score: {score:.2f}", (30, 510), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,128,255), 2)
        else:
            drowsy_counter = 0
            yawn_counter = 0
            blink_counter = 0
            alert_played = False

        cv2.imshow('Driver Monitor', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
