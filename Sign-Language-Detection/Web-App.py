import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pickle
import base64
from PIL import Image
from pathlib import Path

# Must be the first Streamlit command
st.set_page_config(page_title="Sign Language Recognition", layout="centered")

# Load background image and style
def set_background(image_file):
    with open(image_file, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/jpeg;base64,{encoded}") no-repeat center center fixed;
            background-size: cover;
        }}
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.55);
            z-index: -1;
        }}
        /* Set all text color to white */
        h1, h2, h3, h4, h5, h6, p, span, div, label {{
            color: white !important;
        }}
        /* Make only button text black */
        .stButton > button, .stButton > button > div, .stButton > button > div > p {{
            color: black !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set background image
set_background("background1.jpg")  # Replace with your actual image file

# Load trained model
with open('model.p', 'rb') as f:
    model_dict = pickle.load(f)
    model = model_dict['model']

# Label dictionary
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

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

st.title("🤟 Real-time Sign Language Recognition")
st.markdown("Use your webcam to detect and classify hand gestures!")

# App state
if "camera_running" not in st.session_state:
    st.session_state.camera_running = False

# About section
def show_about():
    st.subheader("📘 About This Project")
    st.markdown("""
    This project uses **MediaPipe** to detect hand landmarks and a **Random Forest Classifier** to predict static sign language gestures.
    
    - **Frameworks**: Streamlit, OpenCV, MediaPipe, Scikit-learn  
    - **Model**: RandomForest trained on 42-dimension vectors (21 landmarks × 2)  
    - **Classes**:  
        - Hello  
        - I like you  
        - Victory  
        - Yes  
        - No  
        - Ok  
        - Good Luck  
        - Loser  

    💡 Press "Start" to activate webcam, "Stop" to pause, and "End App" to quit the session.
    """)

# Main buttons
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.button("▶️ Start"):
        st.session_state.camera_running = True
with col2:
    if st.button("⏹️ Stop"):
        st.session_state.camera_running = False
with col3:
    if st.button("📘 About"):
        show_about()

# Display webcam and prediction
frame_display = st.empty()
prediction_display = st.empty()

if st.session_state.camera_running:
    cap = cv2.VideoCapture(0)
    st.success("Webcam is running... Press 'Stop' to pause.")

    while st.session_state.camera_running:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to access webcam.")
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        prediction = "No hand detected"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                x_, y_ = [], []

                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

                if len(data_aux) == 42:
                    pred = model.predict([data_aux])[0]
                    prediction = labels_dict.get(int(pred), str(pred))

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_display.image(frame, channels="RGB", use_column_width=True)
        prediction_display.markdown(f"### ✋ Prediction: `{prediction}`")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
else:
    frame_display.info("Press ▶️ Start to begin webcam detection.")
