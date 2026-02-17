import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import os
import platform
from PIL import Image

# --- Setup MediaPipe Tasks ---
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Model configuration
model_path = 'face_landmarker.task'
IS_WINDOWS = platform.system() == "Windows"

@st.cache_resource
def load_detector():
    if not os.path.exists(model_path):
        # Fallback: Try to download if missing (useful for Cloud)
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
        
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1)
    return vision.FaceLandmarker.create_from_options(options)

# Load detector
try:
    detector = load_detector()
except Exception as e:
    st.error(f"Failed to initialize Mediapipe: {e}")
    st.stop()

# --- App Layout ---
st.set_page_config(page_title="Biometric Face Enrollment Checker", layout="wide")

st.title("🛡️ Biometric Face Enrollment Checker")

# Sidebar settings
st.sidebar.header("⚙️ Settings")
cam_index = st.sidebar.number_input("Camera Index", value=0, step=1)
pose_threshold = st.sidebar.slider("Pose Threshold (Deg)", 10, 45, 25)
light_min = st.sidebar.slider("Min Brightness", 50, 150, 80)
light_max = st.sidebar.slider("Max Brightness", 180, 255, 215)
sharp_min = st.sidebar.slider("Min Sharpness", 20, 200, 70)

st.sidebar.markdown("---")
st.sidebar.info("Tip: If Capture camera doesn't show, make sure Live Camera Toggle is OFF.")

# --- Helper Functions ---
def get_pose(landmarks, img_w, img_h):
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])

    image_points = np.array([
        (landmarks[1].x * img_w, landmarks[1].y * img_h),
        (landmarks[152].x * img_w, landmarks[152].y * img_h),
        (landmarks[33].x * img_w, landmarks[33].y * img_h),
        (landmarks[263].x * img_w, landmarks[263].y * img_h),
        (landmarks[61].x * img_w, landmarks[61].y * img_h),
        (landmarks[291].x * img_w, landmarks[291].y * img_h)
    ], dtype="double")

    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    rmat, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return angles # pitch, yaw, roll

def get_brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def get_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def analyze_frame(frame, status_placeholders):
    h, w, _ = frame.shape
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = detector.detect(mp_image)

    if detection_result.face_landmarks:
        landmarks = detection_result.face_landmarks[0]
        status_placeholders["face"].success("✅ Face Detected")

        brightness = get_brightness(frame)
        if light_min < brightness < light_max:
            status_placeholders["lighting"].success(f"✅ Lighting: OK ({int(brightness)})")
        else:
            status_placeholders["lighting"].error(f"❌ Lighting: {int(brightness)}")

        p, y, r = get_pose(landmarks, w, h)
        if abs(p) < pose_threshold and abs(y) < pose_threshold and abs(r) < pose_threshold:
            status_placeholders["pose"].success("✅ Pose: Straight")
        else:
            status_placeholders["pose"].warning(f"⚠️ Align (P:{int(p)}, Y:{int(y)})")

        eye_dist = (abs(landmarks[159].y - landmarks[145].y) + abs(landmarks[386].y - landmarks[374].y)) / 2
        if eye_dist > 0.011:
            status_placeholders["eyes"].success("✅ Eyes: Open")
        else:
            status_placeholders["eyes"].error("❌ Eyes: Closed")

        sharpness = get_sharpness(frame)
        if sharpness > sharp_min:
            status_placeholders["sharpness"].success(f"✅ Sharpness: OK ({int(sharpness)})")
        else:
            status_placeholders["sharpness"].error(f"❌ Sharpness: Blurry ({int(sharpness)})")
            
        return (light_min < brightness < light_max) and (abs(p) < pose_threshold and abs(y) < pose_threshold and abs(r) < pose_threshold) and (eye_dist > 0.011) and (sharpness > sharp_min)
    else:
        status_placeholders["face"].error("❌ No Face Detected")
        for k in ["lighting", "pose", "eyes", "sharpness"]:
            status_placeholders[k].warning("Waiting...")
        return False

# --- UI Tabs ---
selected_tab = st.sidebar.radio("Navigation", ["🔴 Live Check", "📸 Capture", "📁 Upload"], index=0)

if selected_tab == "🔴 Live Check":
    st.subheader("🔴 Real-time Biometric Diagnostic")
    
    if not IS_WINDOWS:
        st.warning("⚠️ **Live Check** Mode (OpenCV) is mainly for local development on Windows. If you are on Streamlit Cloud or Linux, please use the **Capture** or **Upload** tabs.")
    else:
        st.info("Continuous analysis using local camera. Toggle below to start.")
    
    col_cam, col_info = st.columns([2, 1])
    
    with col_info:
        st.subheader("Diagnostic Stream")
        live_status = {k: st.empty() for k in ["face", "lighting", "pose", "eyes", "sharpness"]}
        overall_live = st.empty()

    with col_cam:
        run_live = st.toggle("Start Camera", value=False, key="live_toggle")
        frame_placeholder = st.empty()
        
        if run_live:
            # Use CAP_DSHOW only on Windows
            cap_flag = cv2.CAP_DSHOW if IS_WINDOWS else cv2.CAP_ANY
            vid = cv2.VideoCapture(int(cam_index), cap_flag)
            if not vid.isOpened():
                st.error(f"Could not open camera at index {cam_index}.")
                run_live = False
            
            # Use a slightly more efficient loop for Streamlit
            while run_live:
                ret, frame = vid.read()
                if not ret: break
                
                is_passed = analyze_frame(frame, live_status)
                if is_passed: overall_live.success("🎯 ENROLLMENT READY")
                else: overall_live.info("⌛ Keep Adjusting...")

                frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
                time.sleep(0.01)
            
            vid.release()
        else:
            frame_placeholder.warning("Camera is OFF.")

elif selected_tab == "📸 Capture":
    st.subheader("📸 Standard Capture Check")
    # Explanation for potential hardware lock
    st.warning("⚠️ If camera fails to load here, ensure the 'Live Check' camera toggle is OFF.")
    
    col_cap, col_info_cap = st.columns([2, 1])
    with col_cap:
        img_file_buffer = st.camera_input("Capture Enrollment Photo")
    with col_info_cap:
        st.subheader("Capture Result")
        cap_status = {k: st.empty() for k in ["face", "lighting", "pose", "eyes", "sharpness"]}
        if img_file_buffer:
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            passed = analyze_frame(cv2_img, cap_status)
            if passed: 
                st.balloons()
                st.success("🎯 Standards Met!")
            else: st.warning("⚠️ Improvement needed.")

elif selected_tab == "📁 Upload":
    st.subheader("📁 Image Upload Service")
    uploaded_file = st.file_uploader("Choose a face image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        col_up, col_info_up = st.columns([2, 1])
        with col_up:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            cv2_up = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        with col_info_up:
            st.subheader("Upload Analysis")
            up_status = {k: st.empty() for k in ["face", "lighting", "pose", "eyes", "sharpness"]}
            passed = analyze_frame(cv2_up, up_status)
            if passed: st.success("🎯 Image Valid for Enrollment")
            else: st.error("❌ Image Fails Standards")

st.divider()
st.caption("Developed for Professional Biometric Enrollment Quality Assurance. (Mediapipe Tasks v2)")
