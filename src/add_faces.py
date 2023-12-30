import cv2
import detect_face
import numpy as np
import pickle
import os

cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    print("Không thể mở camera.")
    exit()

template = cv2.imread('./image/input/template2.png')

name = input("Cho xin cái tên !!!: ")
faces_data = []
count = 0

# read frame and show
while True:
    ret, frame = cap.read()

    if not ret:
        print("Không thể đọc frame.")
        break
        
    flipped_frame = cv2.flip(frame, 1)
    face_detect, face_crop = detect_face.detect_face_with_template(flipped_frame, template)
    mask_img = detect_face.get_ycrcb_mask(flipped_frame)
    
    # add face to data
    if len(faces_data) <= 100 and count % 10 == 0:
        faces_data.append(face_crop)
    count+=1
    cv2.putText(face_detect, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
    
    # show frame
    cv2.imshow('Video1', face_detect)
    cv2.imshow('Video2', face_crop)
    cv2.imshow('Video3', mask_img[1])
    
    k = cv2.waitKey(1)

    if  k==ord('q') or len(faces_data) == 100:
        break

cap.release()
cv2.destroyAllWindows()

# save faces data
faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(100, -1)
print(faces_data.shape)

if __name__ == "__main__":
    pass