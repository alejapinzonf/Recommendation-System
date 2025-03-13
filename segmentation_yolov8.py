import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO(' path')
path = 'path'
cap = cv2.VideoCapture(path)
if not cap.isOpened():
    print("Error al abrir el video")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
videoseg = 'result.mp4'
out = cv2.VideoWriter(videoseg, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    class_frame = results[0].plot()
    cv2.imshow('result', class_frame)
    out.write(class_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
out.release()
cv2.destroyAllWindows()
