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
import matplotlib.pyplot as plt
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# --------------------------------------------
# Streamlit App Configuration
# --------------------------------------------
st.set_page_config(page_title="Cough + Tripod Detector", layout="wide")
os.makedirs("saved_audio", exist_ok=True)

# --------------------------------------------
# Load YAMNet Model
# --------------------------------------------
@st.cache_resource
def load_yamnet():
    """Load YAMNet from TF Hub and return model and class names."""
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
    import pandas as pd
    class_map = pd.read_csv(class_map_path)
    class_names = class_map['display_name'].tolist()
    return yamnet_model, class_names

# --------------------------------------------
# Audio Cough Detection
# --------------------------------------------
def predict_cough_from_file(file_path, yamnet_model, class_names, threshold=0.3):
    y, sr = librosa.load(file_path, sr=None, mono=True)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000
    y = y.astype(np.float32)

    scores, embeddings, spectrogram = yamnet_model(y)
    scores_np = scores.numpy()
    mean_scores = np.mean(scores_np, axis=0)

    cough_idx = next((i for i, name in enumerate(class_names) if "cough" in name.lower()), None)
    cough_prob = float(mean_scores[cough_idx]) if cough_idx is not None else 0.0

    return cough_prob

# --------------------------------------------
# Respiration Rate Estimation
# --------------------------------------------
def calculate_rpm(audio_file, plot=False):
    y, sr = librosa.load(audio_file, sr=None)
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    nyquist = 0.5 * sr
    low = 0.1 / nyquist
    high = 2.0 / nyquist
    b, a = signal.butter(2, [low, high], btype='band')
    y_filtered = signal.filtfilt(b, a, y)

    envelope = np.abs(signal.hilbert(y_filtered))
    peaks, _ = signal.find_peaks(envelope, distance=sr * 0.5)

    duration_min = len(y) / sr / 60.0
    num_breaths = len(peaks)
    rpm = num_breaths / duration_min

    if plot:
        t = np.arange(len(y)) / sr
        plt.figure(figsize=(12, 4))
        plt.plot(t, y, label='Raw signal')
        plt.plot(t, y_filtered, label='Filtered (0.1-2 Hz)')
        plt.plot(t, envelope, label='Envelope', alpha=0.7)
        plt.plot(peaks / sr, envelope[peaks], 'rx', label='Detected breaths')
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title(f"Estimated RPM: {rpm:.2f}")
        plt.legend()
        st.pyplot(plt)
        plt.close()

    return rpm, num_breaths

# --------------------------------------------
# Pose Estimation Utilities
# --------------------------------------------
def angle_2d(a, b):
    return degrees(atan2(b[1] - a[1], b[0] - a[0]))

def distance(a, b):
    return np.hypot(a[0] - b[0], a[1] - b[1])

def evaluate_tripod(landmarks, image_shape):
    """Compute if person is in tripod position using heuristic rules."""
    h, w, _ = image_shape

    def get_landmark(idx):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h])

    # Extract key landmarks
    shoulder = (get_landmark(mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value) +
                get_landmark(mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value)) / 2
    hip = (get_landmark(mp.solutions.pose.PoseLandmark.RIGHT_HIP.value) +
           get_landmark(mp.solutions.pose.PoseLandmark.LEFT_HIP.value)) / 2
    knee_r = get_landmark(mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value)
    knee_l = get_landmark(mp.solutions.pose.PoseLandmark.LEFT_KNEE.value)
    wrist_r = get_landmark(mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value)
    wrist_l = get_landmark(mp.solutions.pose.PoseLandmark.LEFT_WRIST.value)

    # Reference body size (shoulder width)
    shoulder_r = get_landmark(mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value)
    shoulder_l = get_landmark(mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value)
    ref_dist = distance(shoulder_r, shoulder_l) + 1e-6

    # Heuristic 1: torso leaning forward
    torso_angle = angle_2d(hip, shoulder)
    leaning_forward = torso_angle < 60

    # Heuristic 2: sitting posture (hips not far above knees)
    knee_mid = (knee_r + knee_l) / 2
    hip_height = hip[1]
    sitting_like = hip_height < knee_mid[1] * 1.15

    # Heuristic 3: hands near knees/thighs
    dist_r = distance(wrist_r, knee_r) / ref_dist
    dist_l = distance(wrist_l, knee_l) / ref_dist
    hands_near_knees = (dist_r < 1.0 and dist_l < 1.0)

    # Combine score
    score = 0.4 * leaning_forward + 0.3 * sitting_like + 0.3 * hands_near_knees
    is_tripod = score > 0.6

    return {
        "tripod": is_tripod,
        "score": score,
        "leaning_forward": leaning_forward,
        "sitting_like": sitting_like,
        "hands_near_knees": hands_near_knees,
    }

# --------------------------------------------
# Tripod Detection from Video
# --------------------------------------------
def process_video_tripod(input_path, output_path):
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))

    frame_idx = 0
    tripod_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            res = evaluate_tripod(results.pose_landmarks.landmark, frame.shape)

            label = f"Tripod: {'YES' if res['tripod'] else 'NO'}  score={res['score']:.2f}"
            color = (0, 255, 0) if res['tripod'] else (0, 0, 255)
            cv2.rectangle(frame, (5, 5), (450, 55), (0, 0, 0), -1)
            cv2.putText(frame, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if res["tripod"]:
                tripod_frames += 1

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    pose.close()

    tripod_ratio = tripod_frames / max(1, frame_idx)
    return output_path, tripod_ratio

# --------------------------------------------
# Webcam Video Recording Transformer
# --------------------------------------------
class VideoRecorder(VideoTransformerBase):
    def __init__(self):
        self.frames = []

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        self.frames.append(frame.to_ndarray(format="bgr24"))
        return frame

# --------------------------------------------
# Main Streamlit App
# --------------------------------------------
def main():
    st.title("🎤 Cough + Breathing → 🧍 Tripod Detection App")
    yamnet_model, class_names = load_yamnet()

    # --------------------
    # 1️⃣ Audio Detection
    # --------------------
    st.header("Step 1: Record or Upload Your Audio")
    cough_detected = False
    audio_source = st.radio("Choose audio input method:", ["🎙️ Record Audio", "📁 Upload Audio File"])
    save_path = None

    if audio_source == "🎙️ Record Audio":
        st.subheader("🎤 Record your breathing or cough below:")
        audio_bytes = audio_recorder(sample_rate=16000)
        if audio_bytes:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"saved_audio/recording_{timestamp}.wav"
            with open(save_path, "wb") as f:
                f.write(audio_bytes)
            st.audio(save_path, format="audio/wav")
            st.success(f"✅ Recording saved as `{save_path}`")

    elif audio_source == "📁 Upload Audio File":
        st.subheader("📁 Upload your audio file (wav/mp3/m4a/etc.)")
        audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a", "flac", "ogg"])
        if audio_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"saved_audio/uploaded_{timestamp}_{audio_file.name}"
            with open(save_path, "wb") as f:
                f.write(audio_file.read())
            st.audio(save_path, format="audio/wav")

    if save_path:
        with st.spinner("Analyzing audio..."):
            try:
                cough_prob = predict_cough_from_file(save_path, yamnet_model, class_names)
                st.write(f"Predicted cough probability: **{cough_prob:.3f}**")
                cough_detected = cough_prob > 0.3
            except Exception as e:
                st.error(f"Error detecting cough: {e}")

        with st.spinner("Estimating respiration rate..."):
            try:
                rpm, num_breaths = calculate_rpm(save_path)
                st.metric(label="🫁 Estimated Respirations Per Minute", value=f"{rpm:.1f}")
                st.caption(f"Detected {num_breaths} breaths in this recording.")
            except Exception as e:
                st.error(f"Could not estimate RPM: {e}")

        if cough_detected:
            st.success("💨 Cough detected! You can now proceed to video analysis.")
        else:
            st.info("🫁 No strong cough detected. You can still analyze your posture.")

    # --------------------
    # 2️⃣ Tripod Detection
    # --------------------
    st.header("Step 2: Tripod Detection from Video")
    video_source = st.radio("Choose video input method:", ["📁 Upload Video", "🎥 Record with Webcam"])
    video_path = None

    if video_source == "📁 Upload Video":
        video_file = st.file_uploader("Upload a video (mp4/mov/avi/mkv)", type=["mp4", "mov", "avi", "mkv"])
        if video_file:
            vtemp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.name)[1])
            vtemp.write(video_file.read())
            vtemp.close()
            video_path = vtemp.name


    elif video_source == "🎥 Record with Webcam":
        st.info("Click 'Start' to record your webcam. Press 'Stop' and then 'Save Recording'.")
        webrtc_ctx = webrtc_streamer(key="tripod-recorder", video_transformer_factory=VideoRecorder)
        if webrtc_ctx.video_transformer:
            if st.button("Save Recording"):
                frames = webrtc_ctx.video_transformer.frames
                if frames:
                    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                    h, w, _ = frames[0].shape
                    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
                    for frame in frames:
                        writer.write(frame)
                    writer.release()
                    video_path = out_path
                    st.success("🎬 Recording saved successfully!")
                else:
                    st.warning("No frames recorded. Try again.")

    if video_path:
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        with st.spinner("Analyzing video for tripod position..."):
            try:
                out_path, tripod_ratio = process_video_tripod(video_path, out_path)
                st.success(f"🧍 Tripod detected in **{tripod_ratio*100:.1f}%** of frames.")
                st.video(out_path)
            except Exception as e:
                st.error(f"Error processing video: {e}")

if __name__ == "__main__":
    main()
