import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 加载预训练模型
model = load_model("gender_1.h5")

# 加载人脸检测器
face_cascade_front = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_cascade_side = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

# 打开摄像头
cap = cv2.VideoCapture(0)

def remove_overlapping_boxes(boxes, threshold=0.3):
    """
    去除重叠的检测框
    :param boxes: 检测框列表，格式为 [(x, y, w, h), ...]
    :param threshold: 重叠阈值，超过该阈值的检测框会被去除
    :return: 去除重叠后的检测框列表
    """
    # 计算检测框的面积
    areas = [w * h for (x, y, w, h) in boxes]
    # 按面积从大到小排序
    sorted_boxes = [box for _, box in sorted(zip(areas, boxes), key=lambda x: x[0], reverse=True)]
    # 存储最终的检测框
    result_boxes = []
    for box in sorted_boxes:
        x1, y1, w1, h1 = box
        # 检查当前检测框是否与已选检测框重叠
        overlap = False
        for (x2, y2, w2, h2) in result_boxes:
            # 计算交集区域
            ix1 = max(x1, x2)
            iy1 = max(y1, y2)
            ix2 = min(x1 + w1, x2 + w2)
            iy2 = min(y1 + h1, y2 + h2)
            # 计算交集面积
            intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            # 计算并集面积
            union = w1 * h1 + w2 * h2 - intersection
            # 计算重叠比例
            iou = intersection / union
            if iou > threshold:
                overlap = True
                break
        if not overlap:
            result_boxes.append(box)
    return result_boxes

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 转换为灰度图像
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    front_faces = face_cascade_front.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    side_faces = face_cascade_side.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    faces = list(front_faces) + list(side_faces)

    # 去除重叠的检测框
    faces = remove_overlapping_boxes(faces)

    if len(faces) > 0:
        # 遍历每一张检测到的人脸
        for (x, y, w, h) in faces:
            # 扩大人脸区域的范围（比如向上、向左扩展一定的像素）
            expand_pixels = 20  # 扩展的像素数，可以根据需要调整

            # 计算新的边界框
            new_x = max(0, x - expand_pixels)  # 确保不超出图像边界
            new_y = max(0, y - expand_pixels)
            new_w = w + 2 * expand_pixels
            new_h = h + 2 * expand_pixels

            # 提取扩大的人脸区域
            face_roi = gray_image[new_y:new_y + new_h, new_x:new_x + new_w]

            # 检查人脸区域是否有效
            if face_roi.size == 0:
                print("人脸区域提取失败")
                continue

            # 预处理人脸区域
            image_resize = cv2.resize(face_roi, (128, 128))
            image_array = np.expand_dims(image_resize, axis=-1)
            image_array = np.expand_dims(image_array, axis=0)
            image_array = image_array / 255.0

            # 使用模型进行预测
            predictions = model.predict(image_array)
            prediction = predictions[0]
            class_prediction = np.argmax(prediction)
            class_names = ["male", "female"]

            # 打印预测结果
            for i, predict in enumerate(prediction):
                print(f"{class_names[i]}: {predict:.4f}")

            # 在图像上绘制检测框和预测结果
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"{class_names[class_prediction]}: {prediction[class_prediction]:.2f}",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 显示结果
    cv2.imshow("Face Detection with CNN", frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源并关闭窗口
cap.release()
cv2.destroyAllWindows()