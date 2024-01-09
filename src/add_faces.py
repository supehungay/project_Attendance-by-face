import cv2
import detect_face
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from info_to_database import img2db, info2db, train_model_knn, sift_descriptor, key_des2db

def main():
    cap = cv2.VideoCapture(0) 
    sift= cv2.xfeatures2d.SIFT_create(contrastThreshold=0.03, edgeThreshold=10)
    template = cv2.imread('D:\\dulieuD\\Program Language\\Computer_Vision\\FinalExam\\project_Attendance-by-face\\image\\input\\template2.png', 0)

    # name = 'Hung'
    msv = input("Cho cái mã sv: ")
    ten = input("Cho cái tên !!!: ")
    lop = input("Cho cái lớp: ")

    size = 10
    faces_data = []
    count = 0

    # read frame and show
    while True:
        ret, frame = cap.read()

        flipped_frame = cv2.flip(frame, 1)
        
        # face_detect, face_crop = detect_face.detect_face_with_template(flipped_frame, template)
        top_left, bottom_right = detect_face.detect_face_with_template(flipped_frame, template)
        # print((top_left), (bottom_right))
        face_detect = np.copy(flipped_frame)
        cv2.putText(face_detect, f'{str(len(faces_data))} / {size}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        
        if bottom_right is None:
            cv2.putText(face_detect, "get close to the camera", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        elif top_left[0] < 0 or top_left[1] < 0:
            cv2.putText(face_detect, "Move face to center", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        else:
            cv2.rectangle(face_detect, top_left, bottom_right, (0, 0, 255), 2)
            # cv2.drawContours(image, [max_contour], -1, (0, 255, 0), 1)
            face_crop = flipped_frame[(top_left[1]) : (bottom_right[1]), (top_left[0]) : (bottom_right[0]), :]
            # print(len(face_crop))
            # print(face_crop)
            face_crop = cv2.resize(face_crop, (64, 64))
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            mask_img = detect_face.get_ycrcb_mask(flipped_frame)
            
            # add face to data
            if len(faces_data) <= size and count % 10 == 0:
                faces_data.append(face_crop)
            count+=1
            # cv2.putText(face_detect, f'{str(len(faces_data))} / 30', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
            cv2.imshow('Video2', face_crop)
            cv2.imshow('Video3', mask_img[1])
            
        # show frame
        cv2.imshow('Video1', face_detect)
        k = cv2.waitKey(1)

        if  k==ord('q') or len(faces_data) == size:
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(faces_data) < size:
        print('Chưa đủ số lượng ảnh đầu vào')
        exit()

    # save faces data
    faces_data = np.asarray(faces_data)
    faces_data = faces_data.reshape(size, -1)
    # with open(f'./data/{msv}.pkl', 'wb') as f:
    #     pickle.dump(faces_data, f)

    faces_zip = pickle.dumps(faces_data)

    img2db(msv, faces_zip)
    info2db(msv, ten, lop)
    train_model_knn()

if __name__ == "__main__":
    main()