import cv2
import csv
import os
from datetime import datetime

face_cap = cv2.CascadeClassifier("/opt/anaconda3/lib/python3.12/site-packages/cv2/data/haarcascade_frontalface_default.xml")
csv_file = "attendance.csv"
if not os.path.exists(csv_file):
    with open(csv_file, mode = 'w', newline ='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "FaceID"])
face_id = 0
video_cap = cv2.VideoCapture(0)
while True:
    ret, vid_data = video_cap.read()
    if not ret:
        break

    
    col = cv2.cvtColor(vid_data,cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    faces = face_cap.detectMultiScale(
        col, 
        scaleFactor = 1.2,
        minNeighbors= 5,
        minSize = (30,30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    for (x,y, w, h) in faces:
        face_id+=1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.rectangle(vid_data, (x,y), (x+w, y+h), (0,255,0),2)
        with open(csv_file, mode = 'a', newline ='') as f:
            write = csv.writer(f)
            write.writerow([timestamp, f"Face_{face_id}"])
        cv2.putText(vid_data, f"Face_{face_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)
    cv2.imshow('Face Attendance Logger', vid_data)
    if cv2.waitKey(10) == ord("s"):
        break
video_cap.release()