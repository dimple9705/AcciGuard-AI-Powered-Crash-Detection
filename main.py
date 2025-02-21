import cv2
import pandas as pd
import pygame
import tkinter as tk
from tkinter import messagebox
from ultralytics import YOLO
import cvzone

# Initialize pygame mixer
pygame.mixer.init()
# Load beep sound
beep_sound = pygame.mixer.Sound("beep.mp3")

# Initialize tkinter
root = tk.Tk()
root.withdraw()  # Hide the main window

# Function to display alert message
def show_alert_message():
    messagebox.showinfo("Accident Alert", "An accident has been detected!")

model = YOLO('best.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('cr.mp4')

my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count = 0
accident_detected = False
alert_shown = False  # Flag to track if the alert message has been shown

while True:    
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    # Reset the accident_detected flag for each frame
    accident_detected = False
    
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'accident' in c:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
            # Play beep sound only if accident is detected and it's not already playing
            if not accident_detected:
                beep_sound.play()
                accident_detected = True
                # Show alert message only if it hasn't been shown before
                if not alert_shown:
                    show_alert_message()
                    alert_shown = True
        else:    
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
    
    # Reset alert_shown flag if no accidents are detected in the current frame
    if not accident_detected:
        alert_shown = False
        beep_sound.stop()
            
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()  
cv2.destroyAllWindows()
