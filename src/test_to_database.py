import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("D:\\dulieuD\\Program Language\\Computer_Vision\\FinalExam\\project_Attendance-by-face\\serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://attendance-by-face-7de0b-default-rtdb.firebaseio.com/',
    'storageBucket': 'gs://attendance-by-face-7de0b.appspot.com'
})

ref = db.reference('Students')

def info2db(msv, ten, lop):
    data = {
        '20002053':
            {
                'Họ tên': 'Đỗ Mạnh Hùng',
                'Lớp': 'K65A5'
            }
    }
    key, value = data.items()
    ref.child(key).set(value)

    # for key, value in data.items():
    #     ref.child(key).set(value)