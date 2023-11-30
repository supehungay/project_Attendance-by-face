import cv2
import numpy as np

def pick_color_face(image):
    # Chọn màu da khuôn mặt
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Chuyển đổi ảnh sang không gian màu HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Tạo mask cho vùng màu da
    mask = cv2.inRange(hsv_image, lower_skin, upper_skin)

    # Áp dụng morphological transformations để loại bỏ nhiễu và kết hợp vùng màu da
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Tìm contours của vùng màu da
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Chọn vùng có diện tích lớn nhất làm vùng khuôn mặt
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        return x, y, x + w, y + h

    return None

def detect_face_with_template(image, template_path):
    # Đọc ảnh template khuôn mặt
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)

    # Chọn vùng màu da trong ảnh đầu vào
    face_roi = pick_color_face(image)

    if face_roi:
        # Tạo ảnh con của vùng màu da trong ảnh đầu vào
        face_region = image[face_roi[1]:face_roi[3], face_roi[0]:face_roi[2]]

        # Thực hiện template matching
        result = cv2.matchTemplate(face_region, template, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)

        # Tính toán vị trí tuyệt đối của khuôn mặt trong ảnh gốc
        top_left = (face_roi[0] + max_loc[0], face_roi[1] + max_loc[1])
        bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])

        # Vẽ hình chữ nhật xung quanh khuôn mặt
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

        # Hiển thị ảnh
        cv2.imshow('Detected Face', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Không tìm thấy khuôn mặt trong ảnh.")



# Đường dẫn đến ảnh đầu vào
image_path = './uoa1.jpg'

# Đường dẫn đến template khuôn mặt
template_path = './template.png'

# Đọc ảnh đầu vào
input_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
input_image = cv2.resize(input_image, (128, 128))
# Gọi hàm detect_face_with_template
detect_face_with_template(input_image, template_path)