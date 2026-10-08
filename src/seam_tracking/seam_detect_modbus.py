from picamera2 import Picamera2
import cv2
import numpy as np
import minimalmodbus
import time

def nothing(x):
    pass

# Ініціалізація Modbus RTU
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

picam2 = Picamera2()# Ініціалізація камери
video_config = picam2.create_video_configuration(main={"size": (2028, 1520), "format": 'XRGB8888'})
picam2.configure(video_config) #Налаштування камери для відеозйомки
picam2.start() # Запуск камери

# Ядро для морфології
kernel = np.ones((3,3), np.uint8)
kernel_split = np.ones((3, 300), np.uint8)  # вертикальна структура

#Вікна для виведення
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
#cv2.namedWindow("thresah", cv2.WINDOW_NORMAL)
#cv2.namedWindow("opening", cv2.WINDOW_NORMAL)
#cv2.namedWindow("closing", cv2.WINDOW_NORMAL)
#cv2.namedWindow("mask", cv2.WINDOW_NORMAL)

while True:
    frame = picam2.capture_array()
    frame = frame [:, :, :3] # Видаляємо альфа-канал, залишаючи тільки R, G, B
    frame = cv2.bilateralFilter(frame, 3, 100, 50)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([20,40,40])
    upper = np.array([120,255,255])
    mask = cv2.inRange(hsv, lower, upper)

    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    split = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_split) #розділення контурів
    #ret1, thresh = cv2.threshold(closing, 100, 255, cv2.THRESH_BINARY) #пошук порогів по ф-ції BINARY 

    contours, h = cv2.findContours(split, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)

    selected_contours = []
    for contour in range(len(contours)):
        width = cv2.boundingRect(contours[contour])[2]
        p = cv2.arcLength(contours[contour], True) #Периметр контура
        num = cv2.approxPolyDP(contours[contour], 0.15*p, True) #Апроксимація контура
        if width > 200 and len(num) == 2: #детектимо контури, які більші зад. ширини та мають 2 вершини
            selected_contours.append(contours[contour])

    if len(selected_contours) == 2:
        if cv2.boundingRect(selected_contours[0][2]) > cv2.boundingRect(selected_contours[1][2]):
            selected_contours.reverse()
        cv2.drawContours(frame, selected_contours[0], -1, (200, 200,0), 3) #малювання контурів
        cv2.drawContours(frame, selected_contours[1], -1, (200, 200,0), 3) #малювання контурів
        x1, y1, w1, h1 = cv2.boundingRect(selected_contours[0])
        x2, y2, w2, h2 = cv2.boundingRect(selected_contours[1])
        frame = cv2.rectangle(frame, (x1,y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
        frame = cv2.rectangle(frame, (x2,y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)
        frame = cv2.rectangle(frame, (x1+w1-70,y1), (x1 + w1, y1 - 25), (0, 0, 0), -1)
        frame = cv2.rectangle(frame, (x2,y2), (x2 + 70, y2 - 25), (0, 0, 0), -1)
        frame = cv2.circle(frame, (x1+w1, y1+h1//2), 10, (0, 255, 0), -1)
        frame = cv2.circle(frame, (x2, y2+h2//2), 10, (0, 255, 0), -1)
        cv2.putText(frame, "Edge", (x1+w1-70,y1), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, "Edge", (x2,y2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        print("x1:", x1+w1, "y1:", y1+h1//2, "x2:", x2, "y2:", y2+h2//2)
        
        try:
            instrument.write_register(0, x1+w1, functioncode=16)
            instrument.write_register(1, y1+h1//2, functioncode=16)
            instrument.write_register(2, x2, functioncode=16)
            instrument.write_register(3, y2+h2//2, functioncode=16)
        except IOError:
            print("⚠️ Помилка з'єднання (перевір порт, кабель, адресу)")
            

    #cv2.imshow("mask", mask)
    #cv2.imshow("opening", opening)
    #cv2.imshow("closing", closing)
    #cv2.imshow("thresah", thresh)
    cv2.imshow("frame", frame)

    k = cv2.waitKey(1) & 0xFF 
    if k == 27:
        break

cv2.destroyAllWindows()
