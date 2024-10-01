import cv2
import numpy as np
from deepface import DeepFace
import threading
import tkinter as tk
from tkinter import messagebox
import os

class GenderAgeEmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gender, Age, and Emotion Detection")
        self.root.geometry("400x300")

        self.start_button = tk.Button(root, text="Start Detection", command=self.start_detection, height=2, width=20, font=('Helvetica', 12))
        self.start_button.pack(pady=30)

        self.exit_button = tk.Button(root, text="Exit", command=self.exit_application, height=2, width=20, font=('Helvetica', 12))
        self.exit_button.pack(pady=30)

        self.running = False
        self.thread = None
        self.video_writer = None

    def preprocess_image(self, image, size=(227, 227)):
        blob = cv2.dnn.blobFromImage(image, 1.0, size, (104.0, 177.0, 123.0), swapRB=False, crop=False)
        return blob

    def predict_gender_age(self, frame, gender_net, age_net):
        blob = self.preprocess_image(frame)
        gender_net.setInput(blob)
        age_net.setInput(blob)
        gender_preds = gender_net.forward()
        age_preds = age_net.forward()
        gender_list = ['Male', 'Female']
        age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(21-24)', '(25-32)','(33-37)', '(38-43)', '(48-53)', '(60-100)']
        gender = gender_list[gender_preds[0].argmax()]
        age = age_list[age_preds[0].argmax()]
        return gender, age

    def predict_emotion(self, face):
        try:
            analysis = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
            if isinstance(analysis, list):
                analysis = analysis[0]
            return analysis['dominant_emotion']
        except Exception as e:
            print(f"Error in emotion detection: {e}")
            return "Unknown"

    def start_detection(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.run_detection)
            self.thread.start()

    def run_detection(self):
        try:
            age_net = cv2.dnn.readNetFromCaffe('age_deploy.prototxt', 'age_net.caffemodel')
            gender_net = cv2.dnn.readNetFromCaffe('gender_deploy.prototxt', 'gender_net.caffemodel')

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                messagebox.showerror("Error", "Could not open video stream.")
                return

            cv2.namedWindow('Gender, Age, and Emotion Detection', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('Gender, Age, and Emotion Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            fps = 30.0
            output_path = 'output_high_fps.avi'
            self.video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

            while self.running:
                ret, frame = cap.read()
                if not ret:
                    messagebox.showerror("Error", "Failed to capture frame.")
                    break

                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

                for (x, y, w, h) in faces:
                    face = frame[y:y+h, x:x+w]
                    gender, age = self.predict_gender_age(face, gender_net, age_net)
                    emotion = self.predict_emotion(face)
                    label = f"Gender: {gender}, Age: {age}, Emotion: {emotion}"
                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                cv2.imshow('Gender, Age, and Emotion Detection', frame)
                self.video_writer.write(frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            self.video_writer.release()
            cv2.destroyAllWindows()

        except Exception as e:
            self.handle_not_responding(e)

    def handle_not_responding(self, error):
        response = messagebox.askquestion("Error", f"Application not responding: {error}\nWould you like to restart?")
        if response == 'yes':
            self.restart_application()
        else:
            self.exit_application()

    def restart_application(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()
        messagebox.showinfo("Restarting", "Rebooting the application...")
        self.start_detection()

    def exit_application(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = GenderAgeEmotionApp(root)
    root.mainloop()
