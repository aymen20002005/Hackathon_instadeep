import streamlit as st
import tempfile
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import cv2
import mediapipe as mp
from math import atan2, degrees
from datetime import datetime
from audio_recorder_streamlit import audio_recorder
import scipy.signal as signal
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# -----------------------------------------------------
# Streamlit App Configuration & Custom Styling
# -----------------------------------------------------
st.set_page_config(page_title="Cough + Tripod Detector", layout="wide")

st.markdown("""
    <style>
        .stApp {
            background-color: #f9fafb;
            font-family: "Inter", sans-serif;
        }
        h1, h2, h3, h4 {
            font-weight: 700;
            color: #1f2937;
        }
        .section-card {
            background-color: white;
            padding: 1.5rem 2rem;
            border-radius: 1rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            margin-bottom: 2rem;
        }
        .footer {
            text-align: center;
            color: #9ca3af;
            font-size: 0.9em;
            margin-top: 3rem;
        }
    </style>
""", unsafe_allow_html=True)

os.makedirs("saved_audio", exist_ok=True)

# -----------------------------------------------------
# Load YAMNet Model
# -----------------------------------------------------
@st.cache_resource
def load_yamnet():
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
    import pandas as pd
    class_map = pd.read_csv(class_map_path)
    class_names = class_map['display_name'].tolist()
    return yamnet_model, class_names

# -----------------------------------------------------
# Audio Utilities
# -----------------------------------------------------
def predict_cough_from_file(file_path, yamnet_model, class_names, threshold=0.3):
    y, sr = librosa.load(file_path, sr=None, mono=True)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    y = y.astype(np.float32)

    scores, _, _ = yamnet_model(y)
    mean_scores = np.mean(scores.numpy(), axis=0)
    cough_idx = next((i for i, n in enumerate(class_names) if "cough" in n.lower()), None)
    return float(mean_scores[cough_idx]) if cough_idx is not None else 0.0

def calculate_rpm(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    nyquist = 0.5 * sr
    b, a = signal.butter(2, [0.1/nyquist, 2.0/nyquist], btype='band')
    y_filt = signal.filtfilt(b, a, y)
    envelope = np.abs(signal.hilbert(y_filt))
    peaks, _ = signal.find_peaks(envelope, distance=sr * 0.5)
    duration_min = len(y) / sr / 60
    return len(peaks) / duration_min, len(peaks)

# -----------------------------------------------------
# Pose Detection Utilities
# -----------------------------------------------------
def angle_2d(a, b): return degrees(atan2(b[1]-a[1], b[0]-a[0]))
def distance(a, b): return np.hypot(a[0]-b[0], a[1]-b[1])

def evaluate_tripod(landmarks, image_shape):
    h, w, _ = image_shape
    def get_lm(idx):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h])

    mp_pose = mp.solutions.pose
    shoulder = (get_lm(mp_pose.PoseLandmark.RIGHT_SHOULDER.value) + get_lm(mp_pose.PoseLandmark.LEFT_SHOULDER.value)) / 2
    hip = (get_lm(mp_pose.PoseLandmark.RIGHT_HIP.value) + get_lm(mp_pose.PoseLandmark.LEFT_HIP.value)) / 2
    knee_r = get_lm(mp_pose.PoseLandmark.RIGHT_KNEE.value)
    knee_l = get_lm(mp_pose.PoseLandmark.LEFT_KNEE.value)
    wrist_r = get_lm(mp_pose.PoseLandmark.RIGHT_WRIST.value)
    wrist_l = get_lm(mp_pose.PoseLandmark.LEFT_WRIST.value)
    ref = distance(get_lm(mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
                   get_lm(mp_pose.PoseLandmark.LEFT_SHOULDER.value)) + 1e-6

    torso_angle = angle_2d(hip, shoulder)
    leaning = torso_angle < 60
    knee_mid = (knee_r + knee_l) / 2
    sitting = hip[1] < knee_mid[1] * 1.15
    hands_near = (distance(wrist_r, knee_r)/ref < 1 and distance(wrist_l, knee_l)/ref < 1)

    score = 0.4*leaning + 0.3*sitting + 0.3*hands_near
    return {"tripod": score > 0.6, "score": score}

def process_video_tripod(input_path, output_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)
    draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(input_path)
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (w, h))

    total, tripod_frames = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.pose_landmarks:
            draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            eval_res = evaluate_tripod(res.pose_landmarks.landmark, frame.shape)
            label = f"Tripod: {'YES' if eval_res['tripod'] else 'NO'} | score={eval_res['score']:.2f}"
            cv2.rectangle(frame, (10,10), (460,50), (0,0,0), -1)
            cv2.putText(frame, label, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0,255,0) if eval_res['tripod'] else (0,0,255), 2)
            if eval_res['tripod']: tripod_frames += 1
        writer.write(frame)
        total += 1
    cap.release(); writer.release(); pose.close()
    return output_path, tripod_frames / max(total, 1)

class VideoRecorder(VideoTransformerBase):
    def __init__(self): self.frames = []
    def transform(self, f: av.VideoFrame): self.frames.append(f.to_ndarray(format="bgr24")); return f

# -----------------------------------------------------
# Main App
# -----------------------------------------------------
def main():
    # -------------------------------
    # Step 0: Pre-chatbot question
    # -------------------------------
    if "user_feeling" not in st.session_state:
        st.session_state.user_feeling = None

    if st.session_state.user_feeling is None:
        st.markdown("<h1 style='text-align:center;'>🤖 Quick Health Check</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align:center;'>Do you feel well today?</h3>", unsafe_allow_html=True)

        col_yes, col_no = st.columns([1, 1])
        with col_yes:
            if st.button("Yes"):
                st.session_state.user_feeling = "yes"
        with col_no:
            if st.button("No"):
                st.session_state.user_feeling = "no"

    # Stop if user said yes
    if st.session_state.user_feeling == "yes":
        st.success("Great! Stay healthy. 🎉")
        st.stop()

    # Only show the main app if user said NO
    if st.session_state.user_feeling == "no":
        st.markdown("<h1 style='text-align:center;'>🎤 Cough & Breathing + 🧍 Tripod Detector</h1>", unsafe_allow_html=True)
        yamnet_model, class_names = load_yamnet()

        tab_audio, tab_video = st.tabs(["🫁 Audio Analysis", "🎥 Posture Detection"])

        # --- AUDIO TAB ---
        with tab_audio:
            with st.container():
                st.markdown("<div class='section-card'>", unsafe_allow_html=True)
                st.subheader("Step 1 · Audio Analysis")
                st.caption("Upload or record breathing/cough audio to detect cough probability and respiration rate.")
                choice = st.radio("Select input method:", ["📁 Upload Audio File", "🎙️ Record Audio"])
                save_path = None

                col1, col2 = st.columns(2)

                # 📁 Upload File
                with col1:
                    if choice == "📁 Upload Audio File":
                        st.info("Upload your audio clip (wav/mp3/m4a/etc.)")
                        file = st.file_uploader("Choose audio file", type=["wav","mp3","m4a","flac","ogg"])
                        if file:
                            name = f"saved_audio/uploaded_{datetime.now():%Y%m%d_%H%M%S}_{file.name}"
                            open(name, "wb").write(file.read())
                            save_path = name
                            st.audio(save_path)
                            st.success("✅ File uploaded successfully")

                # 🎙️ Record
                with col2:
                    if choice == "🎙️ Record Audio":
                        st.info("Click to start and stop recording.")
                        audio_bytes = audio_recorder(sample_rate=16000)
                        if audio_bytes:
                            name = f"saved_audio/recording_{datetime.now():%Y%m%d_%H%M%S}.wav"
                            open(name, "wb").write(audio_bytes)
                            save_path = name
                            st.audio(save_path)
                            st.success("✅ Recording saved")

                # Analysis
                if save_path:
                    with st.spinner("🔍 Analyzing audio..."):
                        try:
                            cough_prob = predict_cough_from_file(save_path, yamnet_model, class_names)
                            rpm, breaths = calculate_rpm(save_path)
                            colA, colB = st.columns(2)
                            with colA: st.metric("💨 Cough Probability", f"{cough_prob:.2f}")
                            with colB: st.metric("🫁 Respirations / min", f"{rpm:.1f}")
                            st.caption(f"Detected {breaths} breaths in this clip.")
                            st.success("💨 Cough detected!" if cough_prob>0.3 else "🫁 No strong cough detected.")
                        except Exception as e:
                            st.error(f"Error: {e}")
                st.markdown("</div>", unsafe_allow_html=True)

        # --- VIDEO TAB ---
        with tab_video:
            with st.container():
                st.markdown("<div class='section-card'>", unsafe_allow_html=True)
                st.subheader("Step 2 · Tripod Detection")
                st.caption("Upload or record a short video to detect the 'tripod' sitting posture.")
                choice = st.radio("Video Input:", ["📁 Upload Video", "🎥 Record Webcam"])
                video_path = None

                if choice == "📁 Upload Video":
                    file = st.file_uploader("Upload video", type=["mp4","mov","avi","mkv"])
                    if file:
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1])
                        tmp.write(file.read()); tmp.close()
                        video_path = tmp.name

                elif choice == "🎥 Record Webcam":
                    st.info("Start webcam, then stop and click 'Save Recording'.")
                    ctx = webrtc_streamer(key="rec", video_transformer_factory=VideoRecorder)
                    if ctx.video_transformer and st.button("Save Recording"):
                        frames = ctx.video_transformer.frames
                        if frames:
                            out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                            h, w, _ = frames[0].shape
                            writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w,h))
                            for f in frames: writer.write(f)
                            writer.release()
                            video_path = out
                            st.success("🎬 Recording saved")
                        else:
                            st.warning("No frames recorded. Try again.")

                if video_path:
                    with st.spinner("🔍 Analyzing posture..."):
                        try:
                            out, ratio = process_video_tripod(video_path, tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name)
                            st.video(out)
                            st.metric("🧍 Tripod Detected Frames", f"{ratio*100:.1f}%")
                            st.success("Tripod posture detected!" if ratio>0.2 else "Tripod posture rarely detected.")
                        except Exception as e:
                            st.error(f"Error processing video: {e}")
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<p class='footer'>© 2025 Cough + Tripod Detector | Built with Streamlit</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
