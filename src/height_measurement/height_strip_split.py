from picamera2 import Picamera2
import cv2
import numpy as np

# Калібрування
PIXELS_PER_MM = 135 / 20  # ≈ 6.8
BASE_Y = 760  # Y-координата лінії на базі
FRAME_WIDTH = 2028
CENTER_X = FRAME_WIDTH // 2  # 1014
CENTER_MARGIN = 50

# Камера
picam2 = Picamera2()
video_config = picam2.create_video_configuration(main={"size": (2028, 1520), "format": 'XRGB8888'})
picam2.configure(video_config)
picam2.start()

kernel = np.ones((3, 3), np.uint8)
kernel_split = np.ones((3, 100), np.uint8)  # Зменшено з 300 до 150 для виявлення низьких ліній

cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

while True:
    frame = picam2.capture_array()
    frame = frame[:, :, :3]  # видалення альфа-каналу
    frame = cv2.bilateralFilter(frame, 3, 100, 50)
    #frame = cv2.convertScaleAbs(frame, alpha=2, beta=30)  # контраст і яскравість
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([20, 40, 40])
    upper = np.array([120, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    split = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_split)

    # Вирізаємо центральну смугу зображення (100 пікселів по X)
    center_strip = split[:, CENTER_X - CENTER_MARGIN: CENTER_X + CENTER_MARGIN]

    contours, _ = cv2.findContours(center_strip, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if contours:
        # Обираємо тільки найяскравіший (найбільший) контур
        largest_cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_cnt)
        cx = x + w // 2
        cy = y + h // 2

        # Зміщення до глобальних координат
        x_global = x + CENTER_X - CENTER_MARGIN
        cx_global = cx + CENTER_X - CENTER_MARGIN

        cv2.drawContours(frame, [largest_cnt + [CENTER_X - CENTER_MARGIN, 0]], -1, (0, 255, 0), 2)
        cv2.putText(frame, "Brightest", (x_global, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        frame = cv2.circle(frame, (cx_global, cy), 4, (255, 0, 0), -1)

        pixel_shift = abs(cy - BASE_Y)
        height_mm = round(pixel_shift / PIXELS_PER_MM, 2)

        print(f"[INFO] Y: {cy}, зсув: {pixel_shift}px => Висота: {height_mm} мм")

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
