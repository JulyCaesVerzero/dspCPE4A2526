import cv2
import numpy as np
import serial
import time

# ---- SERIAL SETUP ----
ser = serial.Serial('COM3', 9600, timeout=1)  # change 'COM3' to your Arduino port
time.sleep(2)  # wait for serial connection

# ---- CAMERA ----
cap = cv2.VideoCapture(0)  # 0 = default camera

# ---- INITIAL SERVO STATE ----
servo_angle = 90  # middle position
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# ---- COLOR RANGE (HSV) for ORANGE ----
lower_color = np.array([10, 100, 100])
upper_color = np.array([25, 255, 255])
# add second range for orange (wraps HSV space)
lower_color2 = np.array([10, 100, 50])
upper_color2 = np.array([25, 255, 200])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_color, upper_color)
    mask2 = cv2.inRange(hsv, lower_color2, upper_color2)
    mask = mask1 | mask2

    # find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # find largest object
        c = max(contours, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(c)
        obj_center_x = x + w // 2

        # draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.circle(frame, (obj_center_x, y+h//2), 5, (255, 0, 0), -1)

        # calculate error from frame center
        error = obj_center_x - frame_width // 2

        # adjust servo (sensitivity factor)
        if abs(error) > 1:  # tolerance
            if error > 0 and servo_angle > 0:
                servo_angle -= 0.5
            elif error < 0 and servo_angle < 180:
                servo_angle += 0.5

            # send to Arduino
            ser.write(f"{servo_angle}\n".encode())

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
ser.close()
