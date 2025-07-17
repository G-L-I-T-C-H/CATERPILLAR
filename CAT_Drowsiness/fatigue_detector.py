import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

# Indices for left and right eyes landmarks in mediapipe's face mesh
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]

def get_eye_landmarks(landmarks, eye_indices):
    return np.array([[landmarks[idx].x, landmarks[idx].y] for idx in eye_indices])

def eye_aspect_ratio(eye):
    # Compute EAR using 6 points: [p1, p2, p3, p4, p5, p6]
    # EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def get_ear_from_landmarks(landmarks):
    left_eye = get_eye_landmarks(landmarks, LEFT_EYE_IDX)
    right_eye = get_eye_landmarks(landmarks, RIGHT_EYE_IDX)
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    return (left_ear + right_ear) / 2.0
