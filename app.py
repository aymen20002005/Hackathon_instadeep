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
from base64 import b64encode


st.set_page_config(page_title="Cough + Tripod Detector", layout="wide")


@st.cache_resource
def load_yamnet():
	"""Load YAMNet from TF Hub and return model and class names."""
	yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
	class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
	import pandas as pd
	class_map = pd.read_csv(class_map_path)
	class_names = class_map['display_name'].tolist()
	return yamnet_model, class_names


def resample_audio_to_16k(src_path, dst_path):
	y, sr = librosa.load(src_path, sr=None, mono=True)
	if sr != 16000:
		y = librosa.resample(y, orig_sr=sr, target_sr=16000)
		sr = 16000
	librosa.output.write_wav(dst_path, y, sr)


def predict_cough_from_file(file_path, yamnet_model, class_names, threshold=0.3):
	"""Return cough probability (0..1) for the provided file path."""
	# Load and resample to 16k
	y, sr = librosa.load(file_path, sr=None, mono=True)
	if y.ndim > 1:
		y = np.mean(y, axis=1)
	if sr != 16000:
		y = librosa.resample(y, orig_sr=sr, target_sr=16000)
		sr = 16000
	y = y.astype(np.float32)

	# Run YAMNet
	scores, embeddings, spectrogram = yamnet_model(y)
	scores_np = scores.numpy()
	mean_scores = np.mean(scores_np, axis=0)

	# find a class that contains 'cough'
	cough_idx = None
	for i, name in enumerate(class_names):
		if 'cough' in name.lower():
			cough_idx = i
			break

	cough_prob = 0.0
	if cough_idx is not None:
		cough_prob = float(mean_scores[cough_idx])

	return cough_prob


def angle_2d(a, b):
	return degrees(atan2(b[1] - a[1], b[0] - a[0]))


def distance(a, b):
	return np.hypot(a[0] - b[0], a[1] - b[1])


def evaluate_tripod_strict(landmarks, image_shape):
	h, w, _ = image_shape

	def get_landmark(idx):
		lm = landmarks[idx]
		return np.array([lm.x * w, lm.y * h])

	mp_pose = mp.solutions.pose
	shoulder_r = get_landmark(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
	shoulder_l = get_landmark(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
	hip_r = get_landmark(mp_pose.PoseLandmark.RIGHT_HIP.value)
	hip_l = get_landmark(mp_pose.PoseLandmark.LEFT_HIP.value)
	knee_r = get_landmark(mp_pose.PoseLandmark.RIGHT_KNEE.value)
	knee_l = get_landmark(mp_pose.PoseLandmark.LEFT_KNEE.value)
	wrist_r = get_landmark(mp_pose.PoseLandmark.RIGHT_WRIST.value)
	wrist_l = get_landmark(mp_pose.PoseLandmark.LEFT_WRIST.value)

	shoulder = (shoulder_r + shoulder_l) / 2
	hip = (hip_r + hip_l) / 2
	knee_mid = (knee_r + knee_l) / 2

	ref_dist = distance(shoulder_r, shoulder_l) + 1e-6

	torso_angle = abs(angle_2d(hip, shoulder))
	leaning_forward = (35 <= torso_angle <= 60)

	hip_to_knee_ratio = (hip[1] - knee_mid[1]) / ref_dist
	sitting_like = (-0.5 < hip_to_knee_ratio < 0.2)

	dist_r = distance(wrist_r, knee_r) / ref_dist
	dist_l = distance(wrist_l, knee_l) / ref_dist
	hands_near_knees = (dist_r < 0.6 and dist_l < 0.6)

	hands_forward = ((wrist_r[1] > shoulder_r[1]) and (wrist_l[1] > shoulder_l[1]))

	symmetry = abs((wrist_r[1] - wrist_l[1])) < (0.3 * ref_dist)

	# numeric proxies for booleans to combine into score
	score = (0.35 * float(leaning_forward) +
			 0.25 * float(sitting_like) +
			 0.25 * float(hands_near_knees) +
			 0.10 * float(hands_forward) +
			 0.05 * float(symmetry))

	is_tripod = (score > 0.5)

	return {
		"tripod": is_tripod,
		"score": score,
		"leaning_forward": leaning_forward,
		"sitting_like": sitting_like,
		"hands_near_knees": hands_near_knees,
		"hands_forward": hands_forward,
		"symmetry": symmetry,
		"torso_angle": torso_angle,
		"hip_to_knee_ratio": hip_to_knee_ratio,
		"dist_r": dist_r,
		"dist_l": dist_l
	}


def process_video_tripod(input_path, output_path, display_every=200):
	mp_pose = mp.solutions.pose
	mp_drawing = mp.solutions.drawing_utils

	pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)
	cap = cv2.VideoCapture(input_path)
	if not cap.isOpened():
		raise IOError(f"Cannot open video: {input_path}")

	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = cap.get(cv2.CAP_PROP_FPS) or 25
	writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

	frame_idx = 0
	tripod_frames = 0

	while True:
		ret, frame = cap.read()
		if not ret:
			break

		results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
		if results.pose_landmarks:
			mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
			res = evaluate_tripod_strict(results.pose_landmarks.landmark, frame.shape)

			label = f"Tripod: {'YES' if res['tripod'] else 'NO'}  score={res['score']:.2f}"
			color = (0, 255, 0) if res['tripod'] else (0, 0, 255)
			cv2.rectangle(frame, (5, 5), (450, 55), (0, 0, 0), -1)
			cv2.putText(frame, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

			if res['tripod']:
				tripod_frames += 1

		writer.write(frame)

		if frame_idx % display_every == 0:
			st.write(f"Processed frame {frame_idx}")

		frame_idx += 1

	cap.release()
	writer.release()
	pose.close()

	tripod_ratio = tripod_frames / max(1, frame_idx)
	return output_path, tripod_ratio


def main():
	st.title("Cough → Tripod Position Detector")

	st.markdown(
		"Upload a recorded audio file (wav/mp3/etc.). If a cough is detected, upload a video to check for tripod position."
	)

	yamnet_model, class_names = load_yamnet()

	# Section 1: Audio
	st.header("1) Cough detection")
	audio_file = st.file_uploader("Upload recorded audio (wav/mp3/m4a)", type=['wav', 'mp3', 'm4a', 'flac', 'ogg'])

	cough_detected = False
	cough_prob = None

	if audio_file is not None:
		with st.spinner("Analyzing audio for cough..."):
			tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1])
			tfile.write(audio_file.read())
			tfile.flush()
			try:
				cough_prob = predict_cough_from_file(tfile.name, yamnet_model, class_names, threshold=0.3)
			except Exception as e:
				st.error(f"Error processing audio: {e}")
				cough_prob = 0.0
			finally:
				tfile.close()

		st.write(f"Predicted cough probability: {cough_prob:.3f}")
		cough_detected = (cough_prob > 0.3)
		if cough_detected:
			st.success("Cough detected — you can now upload a video for tripod detection.")
		else:
			st.info("No cough detected (or below threshold). You can still upload a video if you want to run tripod detection.")

	# Section 2: Video
	st.header("2) Tripod position detection")
	video_file = st.file_uploader("Upload video (mp4/mov/avi)", type=['mp4', 'mov', 'avi', 'mkv'])

	if video_file is not None:
		if not cough_detected:
			st.warning("No cough was detected earlier. The pipeline will still run the tripod detection if you proceed.")

		with st.spinner("Processing video (this may take a while)..."):
			vtemp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.name)[1])
			vtemp.write(video_file.read())
			vtemp.flush()
			vtemp.close()

			out_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
			out_temp.close()
			try:
				out_path, tripod_ratio = process_video_tripod(vtemp.name, out_temp.name)
			except Exception as e:
				st.error(f"Error processing video: {e}")
				out_path = None
				tripod_ratio = 0.0

		if out_path and os.path.exists(out_path):
			st.success(f"Tripod detected in {tripod_ratio*100:.1f}% of frames")
			st.video(out_path)
		else:
			st.error("Failed to produce output video.")


if __name__ == '__main__':
	main()

