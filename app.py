from ultralytics import YOLO
import cv2
import streamlit as st
from PIL import Image
import tempfile
import os

st.title("🔍 Real-time Object Detection & Tracking")
st.write("Using YOLOv8 - CodeAlpha AI Internship")

# Upload option + Webcam
source = st.radio("Choose input source:", ["Webcam", "Upload Video/Image"])

model = YOLO("yolov8n.pt")  # nano model - fast

if source == "Webcam":
    st.write("Click 'Start' to begin webcam detection")
    if st.button("Start Webcam"):
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            results = model.track(frame, persist=True, conf=0.5)
            annotated_frame = results[0].plot()
            
            # Convert to RGB for Streamlit
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            stframe.image(annotated_frame, channels="RGB", use_column_width=True)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()

else:
    uploaded_file = st.file_uploader("Upload a video or image", type=["mp4", "jpg", "png", "jpeg"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        if uploaded_file.type.startswith("image"):
            results = model.predict(tfile.name, conf=0.5)
            annotated = results[0].plot()
            st.image(annotated, caption="Detected Objects", use_column_width=True)
        else:
            # Video processing
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            out_path = "output.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = model.track(frame, persist=True)
                annotated = results[0].plot()
                out.write(annotated)
                stframe.image(annotated, channels="BGR", use_column_width=True)
            
            cap.release()
            out.release()
            st.success("Processing complete!")
            with open(out_path, "rb") as file:
                st.download_button("Download Output Video", file, file_name="detected_output.mp4")

st.caption("YOLOv8 Object Detection & Tracking")