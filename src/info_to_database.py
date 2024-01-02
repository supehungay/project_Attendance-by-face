import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
import os
import cv2
import numpy as np
import pickle

from sklearn.neighbors import KNeighborsClassifier

cred = credentials.Certificate("../serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://attendance-by-face-default-rtdb.firebaseio.com/',
    'storageBucket': 'attendance-by-face.appspot.com'
})

def img2db(msv, faces_zip):
    # folderPath = './data'
    fileName = f'data/{msv}.pkl'
    bucket = storage.bucket('attendance-by-face.appspot.com')
    blob = bucket.blob(fileName)
    blob.upload_from_string(faces_zip)


def info2db(msv, ten, lop):
    ref = db.reference('Students')
    data = {
        msv:
            {
                'Họ tên': ten,
                'Lớp': lop
            }
    }
    key, value = data.popitem()
    ref.child(key).set(value)
    
def train_model_knn():
    bucket = storage.bucket('attendance-by-face.appspot.com')
    blobs = bucket.list_blobs(prefix='data/')
    all_faces = []
    all_msv = []
    for blob in blobs:
        data_as_bytes = blob.download_as_bytes()
        loaded_data = pickle.loads(data_as_bytes)
        file_name = os.path.basename(blob.name)
        msv = os.path.splitext(file_name)[0]
        
        all_msv.extend([msv] * len(loaded_data))
        all_faces.extend(loaded_data)
        
    FACES = np.asarray(all_faces)    
    LABELS = np.asarray(all_msv)
    print(FACES.shape)
    print(LABELS.shape)
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(FACES, LABELS)
    save_model = pickle.dumps(knn)
    blob = bucket.blob('model/model_knn.pkl')
    blob.upload_from_string(save_model)
    

train_model_knn()
def train_model_cnn(msv, faces_data):
    file_model = f'model/model_knn.pkl'
    bucket = storage.bucket('attendance-by-face.appspot.com')
    blob = bucket.blob(file_model)
    check_exists = blob.exists()
    print(check_exists)
    
    LABELS = [msv] * faces_data.shape[0]
    FACES = faces_data
    
    if check_exists:
        data_as_byte = blob.download_as_bytes()
        old_model = pickle.loads(data_as_byte)
        old_model.fit(FACES, LABELS)
        save_model = pickle.dumps(old_model)
    else:
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(FACES, LABELS)
        save_model = pickle.dumps(knn)
        
    blob.upload_from_string(save_model)
            