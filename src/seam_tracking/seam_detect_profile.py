from picamera2 import Picamera2
import cv2
import numpy as np
import minimalmodbus
import time

# Ініціалізація інструменту
instrument = minimalmodbus.Instrument('/dev/ttySC0', 1)  # Порт та адреса slave

# Налаштування параметрів з'єднання
instrument.serial.baudrate = 9600
instrument.serial.bytesize = 8
instrument.serial.parity   = minimalmodbus.serial.PARITY_NONE
instrument.serial.stopbits = 1
instrument.serial.timeout  = 1  # Таймаут на відповідь

instrument.mode = minimalmodbus.MODE_RTU  # Режим RTU
instrument.clear_buffers_before_each_transaction = True
instrument.debug = True  # Увімкнення режиму відладки

def nothing(x):
    pass

picam2 = Picamera2()  # Ініціалізація камери
video_config = picam2.create_video_configuration(main={"size": (2028, 1520), "format": 'XRGB8888'})
picam2.configure(video_config)  # Налаштування камери для відеозйомки
picam2.start()  # Запуск камери

kernel = np.ones((3, 3), np.uint8)
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

prev_gap_index = None  # Для стабілізації позиції зазору

while True:
    frame = picam2.capture_array()
    frame = frame[:, :, :3]  # Видаляємо альфа-канал, залишаючи тільки R, G, B
    frame = cv2.bilateralFilter(frame, 3, 100, 50)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([20, 40, 140])
    upper = np.array([120, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    edges = cv2.Canny(closing, 50, 150)
    contours, h = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    selected_contours = []
    for contour in range(len(contours)):
        width = cv2.boundingRect(contours[contour])[2]
        p = cv2.arcLength(contours[contour], True)  # Периметр контура
        num = cv2.approxPolyDP(contours[contour], 0.15 * p, True)  # Апроксимація контура
        if width > 500 and len(num) == 2:  # детектимо контури, які більші зад. ширини та мають 2 вершини
            selected_contours.append(contours[contour])
    if len(selected_contours) == 2:
        if cv2.boundingRect(selected_contours[0])[0] > cv2.boundingRect(selected_contours[1])[0]:
            selected_contours.reverse()
        cv2.drawContours(frame, selected_contours[0], -1, (200, 200, 0), 3)
        cv2.drawContours(frame, selected_contours[1], -1, (200, 200, 0), 3)
        x1, y1, w1, h1 = cv2.boundingRect(selected_contours[0])
        x2, y2, w2, h2 = cv2.boundingRect(selected_contours[1])
        frame = cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
        frame = cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)
        frame = cv2.rectangle(frame, (x1 + w1 - 70, y1), (x1 + w1, y1 - 25), (0, 0, 0), -1)
        frame = cv2.rectangle(frame, (x2, y2), (x2 + 70, y2 - 25), (0, 0, 0), -1)
        frame = cv2.circle(frame, (x1 + w1, y1 + h1 // 2), 10, (0, 255, 0), -1)
        frame = cv2.circle(frame, (x2, y2 + h2 // 2), 10, (0, 255, 0), -1)
        cv2.putText(frame, "Edge", (x1 + w1 - 70, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, "Edge", (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        print("x1:", x1 + w1, "y1:", y1 + h1 // 2, "x2:", x2, "y2:", y2 + h2 // 2)
        '''
        try:
            instrument.write_register(0, x1 + w1, functioncode=16)
            instrument.write_register(1, y1 + h1 // 2, functioncode=16)
            instrument.write_register(2, x2, functioncode=16)
            instrument.write_register(3, y2 + h2 // 2, functioncode=16)
        except IOError:
            print("⚠️ Помилка з'єднання (перевір порт, кабель, адресу)")
        '''

    # --- НОВИЙ СТІЙКИЙ АЛГОРИТМ GAP ---
    y_center = frame.shape[0] // 2
    scan_band = frame[y_center - 10:y_center + 10, :, 1]  # зелений канал
    profile = np.mean(scan_band, axis=0)
    profile_smooth = cv2.GaussianBlur(profile.astype(np.float32).reshape(1, -1), (51, 1), 0).flatten()
    center = len(profile_smooth) // 2
    if prev_gap_index is None:
        prev_gap_index = center

    search_range = 150
    l_start = max(prev_gap_index - search_range, 0)
    r_end = min(prev_gap_index + search_range, len(profile_smooth))

    local_profile = profile_smooth[l_start:r_end]
    mid = len(local_profile) // 2
    # абсолютний максимум в усьому вікні
    peak_global = np.argmax(local_profile) + l_start

    # зліва від глобального шукаємо пік з різницею яскравості > threshold
    threshold = 20  # мін. різниця до локального піка
    left_peak = l_start
    for i in range(peak_global - 1, l_start, -1):
        if profile_smooth[peak_global] - profile_smooth[i] > threshold:
            left_peak = i
            break

    # справа від глобального шукаємо другий яскравий пік
    right_peak = r_end - 1
    for i in range(peak_global + 1, r_end):
        if profile_smooth[peak_global] - profile_smooth[i] > threshold:
            right_peak = i
            break

    gap_index = np.argmin(profile_smooth[left_peak:right_peak]) + left_peak

    alpha = 0.1
    max_delta = 5  # максимум сдвигов за кадр
    raw_gap = int(alpha * gap_index + (1 - alpha) * prev_gap_index)
    smoothed_gap = max(prev_gap_index - max_delta, min(prev_gap_index + max_delta, raw_gap))
    prev_gap_index = smoothed_gap

    frame = cv2.line(frame, (left_peak, y_center), (left_peak, y_center - 40), (255, 0, 0), 2)
    frame = cv2.line(frame, (right_peak, y_center), (right_peak, y_center - 40), (255, 0, 0), 2)
    frame = cv2.circle(frame, (smoothed_gap, y_center), 5, (0, 0, 255), -1)
    cv2.putText(frame, "Gap", (smoothed_gap - 20, y_center - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    print(f"[PROF]: left = {left_peak}, right = {right_peak}, gap = {smoothed_gap}, width = {right_peak - left_peak}")

    cv2.imshow("frame", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
