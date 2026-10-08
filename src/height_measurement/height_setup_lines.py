from picamera2 import Picamera2
import cv2
import numpy as np

# Ініціалізація камери
picam2 = Picamera2()
video_config = picam2.create_video_configuration(main={"size": (2028, 1520), "format": 'XRGB8888'})
picam2.configure(video_config)
picam2.start()

# Морфологічні ядра
kernel = np.ones((3,3), np.uint8)
kernel_split = np.ones((3, 300), np.uint8)  # вертикальне ядро для розділення лазерної лінії

cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

while True:
    frame = picam2.capture_array()
    frame = frame[:, :, :3]  # Видаляємо альфа-канал (XRGB → RGB)

    # Обробка
    frame = cv2.bilateralFilter(frame, 3, 100, 50)
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    lower = np.array([20, 40, 40])   # підлаштуй під свій лазер
    upper = np.array([120, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    split = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_split)

    contours, _ = cv2.findContours(split, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]  # Тільки 3 найбільші

    y_coords = []

    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        y_center = y + h // 2
        y_coords.append(y_center)

        color = (0, 255, 0) if i == 1 else (0, 255, 255)  # центральна лінія — зелена, краї — жовті
        cv2.line(frame, (0, y_center), (frame.shape[1], y_center), color, 2)
        cv2.putText(frame, f"Y{i+1}: {y_center}", (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    # Сортуємо Y-координати по висоті (згори донизу) для стабільного виводу
    if len(y_coords) == 3:
        y_coords_sorted = sorted(y_coords)
        print(f"🟡 Край 1 (верх): {y_coords_sorted[0]}, 🟢 Центр: {y_coords_sorted[1]}, 🟡 Край 2 (низ): {y_coords_sorted[2]}")

    # Показ зображення
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
