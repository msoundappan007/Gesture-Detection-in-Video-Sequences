# Gesture Detection in Video Sequences

This Streamlit application detects a specified gesture in a video sequence using a pre-trained MobileNet model. The gesture can be provided as an image or video, and the application will annotate the frames in the test video where the gesture is detected.

## Features

- Upload a gesture representation (image or video)
- Upload a test video
- Detect and annotate frames in the test video where the gesture is present
- Display the processed video frames with annotations
- Show the count of frames where the gesture is detected

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/msoundappan007/gesture-detection-video_sequences.git
    cd gesture-detection-video_sequences
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit application:
    ```sh
    streamlit run app.py
    ```

2. Open your web browser and go to `http://localhost:8501`.

3. In the sidebar, upload a gesture representation (either an image or a video).

4. Upload a test video in which you want to detect the gesture.

5. Click the "Detect Gesture" button to start the detection process.

6. The annotated video frames will be displayed, and the count of detected gesture frames will be shown in the sidebar.

## Requirements

- Python 3.7 or higher
- `streamlit`
- `opencv-python-headless`
- `numpy`
- `tensorflow`

## File Structure

- `app.py`: The main application script.
- `requirements.txt`: The dependencies required to run the application.
- `README.md`: This file.



