import streamlit as st
import cv2
import numpy as np
import tempfile
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Load the pre-trained MobileNet model
model = MobileNet(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

def extract_features(image, model):
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = model.predict(image)
    return features.flatten()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def process_gesture_input(uploaded_file, model):
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4' if uploaded_file.type == 'video/mp4' else '.jpg') as tmpfile:
            tmpfile.write(uploaded_file.read())
            input_path = tmpfile.name

        if input_path.endswith('.mp4'):
            cap = cv2.VideoCapture(input_path)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise ValueError("Could not read the video file.")
            return extract_features(frame, model)
        else:
            image = cv2.imread(input_path)
            if image is None:
                raise ValueError("Could not read the image file.")
            return extract_features(image, model)
    else:
        return None

def annotate_and_display_video(test_video_path, gesture_features, model):
    cap = cv2.VideoCapture(test_video_path)
    stframe = st.empty()
    threshold = 0.6
    frame_count = 0
    detected_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_features = extract_features(frame, model)
        similarity = cosine_similarity(gesture_features, frame_features)

        if similarity > threshold:
            detected_count += 1
            cv2.putText(frame, 'DETECTED', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB", use_column_width=True)

    cap.release()

    st.write(f"Processed {frame_count} frames, detected gesture in {detected_count} frames.")
    return detected_count

def main():
    st.title("Gesture Detection in Video Sequences")

    st.sidebar.title("Options")
    st.sidebar.markdown("---")
    gesture_file = st.sidebar.file_uploader("Upload the gesture representation (image or video):", type=['jpg', 'mp4'])
    test_video_file = st.sidebar.file_uploader("Upload the test video:", type='mp4')
    detected_gesture_count = 0

    if st.sidebar.button("Detect Gesture", key="detect_button"):
        if gesture_file and test_video_file:
            gesture_features = process_gesture_input(gesture_file, model)

            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
                tmpfile.write(test_video_file.read())
                test_video_path = tmpfile.name

            detected_gesture_count = annotate_and_display_video(test_video_path, gesture_features, model)
        else:
            st.sidebar.error("Please upload both the gesture representation and the test video.")

    st.sidebar.markdown("---")
    st.sidebar.write(f"Detected gesture count: {detected_gesture_count}")

if __name__ == "__main__":
    main()




