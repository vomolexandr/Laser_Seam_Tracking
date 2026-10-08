from picamera2 import Picamera2
import cv2
import numpy as np

# Калібрування
PIXELS_PER_MM = 135 / 20
BASE_Y = 760  # Y-координата лінії без об'єкта

# Камера
picam2 = Picamera2()
video_config = picam2.create_video_configuration(main={"size": (2028, 1520), "format": 'XRGB8888'})
picam2.configure(video_config)
picam2.start()

cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

while True:
    frame = picam2.capture_array()
    frame = frame[:, :, :3]

    # Фільтрація
    frame = cv2.bilateralFilter(frame, 3, 100, 50)

    # HSV і маска лазера
    #frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=20) #підсилення контрасту та яскравості
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([20, 40, 80])
    upper = np.array([120, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Краї + розширення
    edges = cv2.Canny(mask, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)

    # Контури
    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

    if len(contours) == 3:
        line_centers = []
        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w // 2
            cy = y + h // 2
            line_centers.append((cx, cy, cnt))  # координата X для сортування

        # Сортуємо лінії за X (зліва направо)
        line_centers.sort(key=lambda tup: tup[0])

        # Візуалізація + маркування
        for i, (cx, cy, cnt) in enumerate(line_centers):
            cv2.drawContours(frame, cnt, -1, (0, 255, 0), 2)
            cv2.putText(frame, f"L{i+1}", (cx - 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            frame = cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

        # Вибираємо: центральна лінія — об'єкт, крайня — базова
        y_tip = line_centers[1][1]  # середня по X — над зразком
        y_base = (line_centers[0][1] + line_centers[2][1]) / 2

        pixel_shift = abs(y_tip - y_base)
        height_mm = round(pixel_shift / PIXELS_PER_MM, 2)

        print(" Координати Y ліній:")
        for idx, (_, cy, _) in enumerate(line_centers):
            print(f"  Лінія {idx+1}: Y = {cy}")
        print(f" Зсув: {pixel_shift} пікселів => Висота: {height_mm} мм")

    else:
        print("[WARN] Виявлено менше трьох ліній!")

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
