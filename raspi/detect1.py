import serial
#import threading
import cv2
import numpy as np

import time
from SendData import Send_data
from raspi_models import ModelosCNN

def tiva_leds(msg):
    global msg_prev
    try:
        if msg != msg_prev:
            if msg == 1:
                enviar = 'c\n'
                soda_brands['Coca-cola'] +=1
            elif msg == 2:
                enviar = 'p\n'
                soda_brands['Pepsi'] +=1
            elif msg == 3:
                enviar = 'f\n'
                soda_brands['Fanta'] +=1
            elif msg == 0:
                enviar = 'N\n'
            tiva.write(enviar.encode('utf-8'))
            print(f"Enviado: {enviar}\n")
            msg_prev = msg 
    except:
        print(f"Valor no válido: {msg}")
        msg_prev = msg

def tiva_leds_count(msg):
    global msg_prev
    try:
        if msg != msg_prev:
            if msg == 1:
                enviar = '1\n'
            elif msg == 2:
                enviar = '2\n'
            elif msg == 3:
                enviar = '3\n'
            elif msg == 4:
                enviar = '4\n'
            tiva.write(enviar.encode('utf-8'))
            print(f"Enviado: {enviar}\n")
            msg_prev = msg 
    except:
        print(f"Valor no válido: {msg}")

def count_bottles_move(frame):
    global fgbg
    fgmask = fgbg.apply(frame)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel_open)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    fgmask = cv2.dilate(fgmask, kernel_dilate, iterations=2)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return fgmask, contours

def detectar_color(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([130, 255, 255])
    lower_orange = np.array([10, 100, 100])
    upper_orange = np.array([25, 255, 255])

    mask_red = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1),
                              cv2.inRange(hsv, lower_red2, upper_red2))
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)

    red_pixels = cv2.countNonZero(mask_red)
    blue_pixels = cv2.countNonZero(mask_blue)
    orange_pixels = cv2.countNonZero(mask_orange)

    max_pixels = max(red_pixels, blue_pixels, orange_pixels,1000)
    if max_pixels == 1000:
        return 0
    if max_pixels == red_pixels:
        return 1  # rojo → 'c'
    elif max_pixels == blue_pixels:
        return 2  # azul → 'p'
    elif max_pixels == orange_pixels:
        return 3  # naranja → 'f'

###############Variables
msg_prev = 0
modelos = ModelosCNN(4)
CONFIDENCE_THRESHOLD = 0.8
labels = ['Coca', 'Pepsi', 'Fanta']
flag_count = False
framesPerCount = 3
frameCount = 0
t = -1
mode = 0
Nmodes = 4 #0 cnn, 1 counding boxes static objects, 2 by color, 3 moving objects
flag_change = False
tiempo_medicion = 600 #segundos

class FakeSerial:
    def write(self, data):
        print(f"[FAKE UART] Enviando: {data.decode().strip()}")
    def reset_input_buffer(self):
        print("[FAKE UART] Buffer reseteado")

try:
    tiva = serial.Serial("/dev/ttyACM0", 115200)
    tiva.reset_input_buffer()
except:
    print("Tiva no conectada, usando puerto serial simulado.")
    tiva = FakeSerial()

soda_brands = {
    "Coca-cola": 0,
    "Pepsi": 0,
    "Fanta": 0
}

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
while not cap.isOpened():
    print("Esperando cámara...")
    time.sleep(0.5)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    key = cv2.waitKey(1) & 0xFF
    if key == ord('0'):
        mode = (mode+1)%Nmodes
        flag_change = True

    if mode == 0: #detect bottles with cnn
        if flag_change:
            print("Detect bottles")
            flag_change = False
            cv2.destroyAllWindows()
        frameCount = (frameCount+1) % framesPerCount
        cv2.imshow("Camara", frame)
        if key == ord('s'):
            print("Starting the bottle count (1 min!) ")
            start_time = time.time()
            flag_count = True
            frameCount = 0
        if flag_count:
            t = time.time() - start_time        
        if flag_count and t<tiempo_medicion and frameCount==0:
            predictionArray, predictionText, predictionConf = modelos.Predict(frame)
            if predictionConf < CONFIDENCE_THRESHOLD:
                label = f"nr. Confianza: {predictionConf:.2f}, botella: {predictionText}"
                msg = 0
            else:
                label = f"Confianza: {predictionConf:.2f} botella: {predictionText}"
                msg = np.argmax(predictionArray)+1
                tiva_leds(msg)
                time.sleep(0.2)
            cv2.putText(frame, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow("Predict", frame)    
        elif t>tiempo_medicion:
            flag_count=False
            t = -1
            sender = Send_data("Andy", soda_brands)

    elif mode == 1: #Detect bounding boxes on static objects
        if flag_change:
            print("Detect bounding boxes")
            flag_change = False
            cv2.destroyAllWindows()
        frame_copy = frame.copy()
        gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        opened = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]

        for cnt in filtered_contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame_copy, f"Area: {int(area)}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        print(f"Objetos detectados: {len(filtered_contours)}")
        cv2.imshow("boundin", frame_copy)
        cv2.imshow("opened", opened)

    elif mode == 2: #detect "bottle" by color
        if flag_change:
            print("Detect color")
            flag_change = False
            cv2.destroyAllWindows()
        color_detectado = detectar_color(frame)
        tiva_leds(color_detectado)
        tiva_leds_count(color_detectado)
        time.sleep(0.2)
        cv2.imshow("Detect Color", frame)

    elif mode == 3: #detect movement
        if flag_change:
            print("Detect color")
            flag_change = False
            fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=50, detectShadows=False)
            cv2.destroyAllWindows()  

        # Filtrar contornos
        fgmask, contours = count_bottles_move(frame)
        contours = list(filter(lambda c: cv2.contourArea(c) > 1500, contours))

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(frame, f"Objetos: {len(contours)}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if contours != 0:
            tiva_leds_count(len(contours))
            time.sleep(0.2)
        #cv2.imshow("Frame", frame)
        cv2.imshow("Mask", fgmask)

#######################
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
