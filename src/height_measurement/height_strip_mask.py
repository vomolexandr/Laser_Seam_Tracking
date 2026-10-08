from picamera2 import Picamera2
import cv2
import numpy as np

PIXELS_PER_MM = 135 / 20
BASE_Y = 760
FRAME_WIDTH = 2028
CENTER_X = FRAME_WIDTH // 2
CENTER_MARGIN = 50

picam2 = Picamera2()
video_config = picam2.create_video_configuration(main={"size": (2028, 1520), "format": 'XRGB8888'})
picam2.configure(video_config)
picam2.start()

kernel = np.ones((3, 3), np.uint8)
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

while True:
    frame = picam2.capture_array()
    frame = frame[:, :, :3]
    frame = cv2.bilateralFilter(frame, 3, 100, 50)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([20, 40, 40])
    upper = np.array([120, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    # Обрізаємо центральну вертикальну смугу
    center_strip = closing[:, CENTER_X - CENTER_MARGIN:CENTER_X + CENTER_MARGIN]

    # Малюємо жовтий прямокутник — зону вимірювання
    frame = cv2.rectangle(
        frame,
        (CENTER_X - CENTER_MARGIN, 0),
        (CENTER_X + CENTER_MARGIN, frame.shape[0]),
        (0, 255, 255), 2
    )

    # Знаходимо всі яскраві точки в смузі (тобто частини лінії)
    ys, xs = np.where(center_strip > 0)

    if len(ys) > 0:
        # Обчислюємо верхній, нижній край і центр
        top_y = np.min(ys)
        bottom_y = np.max(ys)
        cy = (top_y + bottom_y) // 2

        cx_global = CENTER_X
        cy_global = cy

        # Малюємо межі лінії і центр
        frame = cv2.line(frame, (CENTER_X - CENTER_MARGIN, top_y), (CENTER_X + CENTER_MARGIN, top_y), (255, 0, 0), 1)
        frame = cv2.line(frame, (CENTER_X - CENTER_MARGIN, bottom_y), (CENTER_X + CENTER_MARGIN, bottom_y), (255, 0, 0), 1)
        frame = cv2.circle(frame, (cx_global, cy_global), 5, (0, 255, 255), -1)
        cv2.putText(frame, "Center", (cx_global + 10, cy_global), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        # Висота
        pixel_shift = abs(cy - BASE_Y)
        height_mm = round(pixel_shift / PIXELS_PER_MM, 2)
        print(f"[INFO] Y: {cy}, зсув: {pixel_shift}px => Висота: {height_mm} мм")
    else:
        print("[WARN] Не знайдено лінії у центрі")

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
