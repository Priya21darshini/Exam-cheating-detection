# Exam-cheating-detection-using-AI
This project is an AI-driven exam cheating detection system using computer vision. It leverages YOLOv8 and head pose estimation to identify abnormal head movements and phone usage. Normal forward-facing posture is marked Non-Cheating, while looking around or using a phone is flagged as Cheating.
Project Overview


This project presents an intelligent surveillance system that automatically monitors students during exams. Traditional human invigilation is inefficient, error-prone, and resource-intensive.
Our solution uses deep learning–based head-pose estimation, eye-gaze tracking, and action recognition to detect suspicious behaviors in real-time.

The tool is designed for:

Examination halls
Online proctored exams
University/classroom surveillance
Research in behavior detection

 Clone the repository
git clone https://github.com/Priya21darshini/Exam-cheating-detection.git
cd Exam-cheating-detection


 Create Virtual Environment
Copy code
python -m venv venv

 Download YOLO Model
pip install ultralytics

Running the Application
streamlit app.py


