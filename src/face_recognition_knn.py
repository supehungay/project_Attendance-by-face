import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
import detect_face

cred = credentials.Certificate("D:\\dulieuD\\Program Language\\Computer_Vision\\FinalExam\\project_Attendance-by-face\\serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://attendance-by-face-7de0b-default-rtdb.firebaseio.com/',
    'storageBucket': 'attendance-by-face-7de0b.appspot.com'
})

file_model = f'model/model_knn.pkl'
bucket = storage.bucket('attendance-by-face-7de0b.appspot.com')
blob = bucket.blob(file_model)
data_as_byte = blob.download_as_bytes()
knn = pickle.loads(data_as_byte)


cap = cv2.VideoCapture(0) 
template = cv2.imread('D:\\dulieuD\\Program Language\\Computer_Vision\\FinalExam\\project_Attendance-by-face\\image\\input\\template2.png', 0)

while True:
    ret, frame = cap.read()

    flipped_frame = cv2.flip(frame, 1)
    top_left, bottom_right = detect_face.detect_face_with_template(flipped_frame, template)
    face_detect = np.copy(flipped_frame)
    
    if bottom_right is None:
        cv2.putText(face_detect, "get close to the camera", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
    elif top_left[0] < 0 or top_left[1] < 0:
        cv2.putText(face_detect, "Move face to center", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
    else:
        cv2.rectangle(face_detect, top_left, bottom_right, (0, 0, 255), 2)
        face_crop = flipped_frame[(top_left[1]) : (bottom_right[1]), (top_left[0]) : (bottom_right[0]), :]
        face_crop = cv2.resize(face_crop, (64, 64))
        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        
        output = knn.predict(face_crop.flatten().reshape(1,-1))
        cv2.putText(face_detect, str(output), top_left, cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        
    # show frame
    cv2.imshow('Video1', face_detect)
    
    k = cv2.waitKey(1)

    if  k==ord('q'):
        break


# with open('./data/20002053.pkl', 'rb') as w:
#     FACES=pickle.load(w)



