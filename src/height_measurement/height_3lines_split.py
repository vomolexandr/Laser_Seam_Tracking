from picamera2 import Picamera2
import cv2
import numpy as np

# Калібрування
PIXELS_PER_MM = 135 / 20  # ≈ 6.8
BASE_Y = 760  # Y-координата лінії на базі

# Камера
picam2 = Picamera2()
video_config = picam2.create_video_configuration(main={"size": (2028, 1520), "format": 'XRGB8888'})
picam2.configure(video_config)
picam2.start()

kernel = np.ones((3,3), np.uint8)
kernel_split = np.ones((3, 300), np.uint8)

cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

while True:
    frame = picam2.capture_array()
    frame = frame[:, :, :3]  # видалення альфа-каналу
    frame = cv2.bilateralFilter(frame, 3, 100, 50)
    #frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=20) #контраст і яскравість
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([20, 40, 60])
    upper = np.array([120, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    split = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_split)

    contours, _ = cv2.findContours(split, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

    if len(contours) == 3:
        line_centers = []
        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            cy = y + h // 2
            line_centers.append(cy)
            cv2.drawContours(frame, cnt, -1, (0, 255, 0), 2)
            cv2.putText(frame, f"Line {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            frame = cv2.circle(frame, (x + w // 2, cy), 5, (255, 0, 0), -1)

        # Важливо: беремо першу лінію як ту, що рухається
        y_line1 = line_centers[1]  # зсувна
        y_line2 = (line_centers[2] + line_centers[2]) / 2 # стабільна

        pixel_shift = abs(y_line1 - BASE_Y)
        height_mm = round(pixel_shift / PIXELS_PER_MM, 2)

        print(f"[INFO] Y лінії 1: {y_line1}, зсув: {pixel_shift} пікселів => Висота: {height_mm} мм")

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
