import tkinter as tk
from tkinter import messagebox, ttk
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pandas as pd

# ========= STEP 1: Load Images ==========
path = 'images'
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print("Encoding Complete ✅")

# ========= STEP 2: Attendance Function ==========
def markAttendance(name):
    file = 'Attendance.csv'
    if not os.path.exists(file):
        df = pd.DataFrame(columns=["Name", "Date", "Time"])
        df.to_csv(file, index=False)

    df = pd.read_csv(file)
    now = datetime.now()
    date = now.strftime('%Y-%m-%d')
    time = now.strftime('%H:%M:%S')

    if not ((df['Name'] == name) & (df['Date'] == date)).any():
        new_row = {"Name": name, "Date": date, "Time": time}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(file, index=False)

# ========= STEP 3: Face Recognition Function ==========
def startRecognition():
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.rectangle(img, (x1,y2-35), (x2,y2), (0,255,0), cv2.FILLED)
                cv2.putText(img, name, (x1+6,y2-6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                markAttendance(name)

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ========= STEP 4: View Attendance ==========
def viewAttendance():
    file = 'Attendance.csv'
    if not os.path.exists(file):
        messagebox.showinfo("No Data", "Attendance file not found!")
        return

    df = pd.read_csv(file)

    top = tk.Toplevel()
    top.title("Attendance Records")
    top.geometry("500x400")

    tree = ttk.Treeview(top)
    tree["columns"] = list(df.columns)
    tree["show"] = "headings"

    for col in df.columns:
        tree.heading(col, text=col)
        tree.column(col, width=100)

    for index, row in df.iterrows():
        tree.insert("", "end", values=list(row))

    tree.pack(fill="both", expand=True)

# ========= STEP 5: Tkinter GUI ==========
root = tk.Tk()
root.title("Face Recognition Attendance System")
root.geometry("400x250")

label = tk.Label(root, text="Attendance System", font=("Arial", 20))
label.pack(pady=20)

start_btn = tk.Button(root, text="Start Face Recognition", command=startRecognition, font=("Arial", 14), bg="green", fg="white")
start_btn.pack(pady=10)

view_btn = tk.Button(root, text="View Attendance", command=viewAttendance, font=("Arial", 14), bg="blue", fg="white")
view_btn.pack(pady=10)

exit_btn = tk.Button(root, text="Exit", command=root.destroy, font=("Arial", 14), bg="red", fg="white")
exit_btn.pack(pady=10)

root.mainloop()
