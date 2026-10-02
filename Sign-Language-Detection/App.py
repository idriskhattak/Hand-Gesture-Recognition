import cv2
import pickle
import numpy as np
import mediapipe as mp

# Load trained model
with open('model.p', 'rb') as f:
    model_dict = pickle.load(f)
    model = model_dict['model']

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

# ✅ Correct class labels
labels_dict = {
    0: 'Hello',
    1: 'I like you',
    2: 'Victory',
    3: 'Yes',
    4: 'No',
    5: 'Ok',
    6: 'Good Luck',
    7: 'Loser'
}

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    h, w, _ = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    prediction = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []

            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            if len(data_aux) == 42:
                pred = model.predict([data_aux])[0]
                label = labels_dict.get(int(pred), str(pred))
                cv2.putText(frame, f'Prediction: {label}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1.3, (0, 255, 0), 3, cv2.LINE_AA)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Sign Language Prediction", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
