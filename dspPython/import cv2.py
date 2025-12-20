import cv2
import serial
import time
import os

# --- 1. HARDWARE SETUP ---
PORT = 'COM3' 
BAUD = 115200
esp32 = None

try:
    esp32 = serial.Serial(PORT, BAUD, timeout=1)
    print(f"✅ ESP32 Linked on {PORT}")
    time.sleep(2) 
except Exception as e:
    print(f"❌ Serial Error: {e}")

# --- 2. FACE DETECTION SETUP ---
# This uses the built-in OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

door_open = False

print("🚀 System Active. Show your face to the camera!")

try:
    while True:
        ret, frame = cap.read()
        if not ret: break

        # Convert to grayscale for the detector
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            if not door_open:
                print("🔓 FACE DETECTED: Opening Door")
                if esp32: esp32.write(b'1')
                door_open = True
            
            # Draw a box around every face found
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            if door_open:
                print("🔒 NO FACE: Locking Door")
                if esp32: esp32.write(b'0')
                door_open = False

        # Display the result
        cv2.imshow('Face Recognition Door Lock', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("Shutting down...")
    if esp32:
        esp32.write(b'0')
        esp32.close()
    cap.release()
    cv2.destroyAllWindows()