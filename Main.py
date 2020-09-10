from tkinter import *
from tkinter import messagebox
import numpy as np
from PIL import Image
import pickle
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import playsound
import argparse
import imutils
import time
import dlib
import cv2
import os

def main():


    def login():
        faces_train()
        time.sleep(1)
        print("training data")
        try:
            x = faces()
            print(type(x))
            print("in login function", x)
            messagebox.showinfo("welcome", x)
            x = print(str(x))
            time.sleep(2)
            if not x:
                time.sleep(2)
                screen.destroy()
                screen1 = Tk()
                screen1.geometry("600x480")
                screen1.title("Authorised")
                screen1.configure(background='turquoise')
                Label(screen1, text="ENGINE START ", fg='purple', bg='turquoise', font=('comicsans', 10)).pack()
                Button(screen1, text="START", height="2", width="20", fg='white', bg='purple', font=('comicsans', 12),
                       command=sleep).place(x=70, y=100)

                screen1.mainloop()
            else:
                messagebox.showinfo("UnAuthorised","Access not given")
                exit()
        except:
            messagebox.showinfo("FACE NOT FOUND","FACE NOT FOUND")
            exit()

    def faces_train():
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        image_dir = os.path.join(BASE_DIR, "image")

        face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
        recognizer = cv2.face.LBPHFaceRecognizer_create()

        current_id = 0
        label_ids = {}
        y_labels = []
        x_train = []

        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.endswith("png") or file.endswith("jpg"):
                    path = os.path.join(root, file)
                    label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
                    # print(label,path)
                    if not label in label_ids:
                        label_ids[label] = current_id
                        current_id += 1
                    id_ = label_ids[label]
                    # print(label_ids)
                    # y_labels.append(label)
                    # x_train.append(path)
                    pil_image = Image.open(path).convert("L")
                    size = (550, 550)
                    final_image = pil_image.resize(size, Image.ANTIALIAS)
                    image_array = np.array(final_image, "uint8")
                    # print(image_array)
                    faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

                    for (x, y, w, h) in faces:
                        roi = image_array[y:y + h, x:x + w]
                        x_train.append(roi)
                        y_labels.append(id_)

        # print(y_labels)
        # print(x_train)

        with open("labels.pickle", 'wb') as f:
            pickle.dump(label_ids, f)

        recognizer.train(x_train, np.array(y_labels))
        recognizer.save("trainner.yml")


    def faces():
        face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("trainner.yml")

        labels = {}
        with open("labels.pickle", 'rb') as f:
            og_labels = pickle.load(f)
            labels = {v: k for k, v in og_labels.items()}

        cap = cv2.VideoCapture(0)

        while (True):
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
            for (x, y, w, h) in faces:
                # print(x,y,w,h)
                roi_gray = gray[y:y + h, x:x + w]
                #roi_color = gray[y:y + h, x:x + w]

                id_, conf = recognizer.predict(roi_gray)
                if conf >= 75 and conf <= 95:
                    # print(id_)
                    print(labels[id_])
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    name = labels[id_]
                    color = (255, 0, 0)
                    stroke = 2
                    cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

                #img_item = "my-image.png"
                #cv2.imwrite(img_item, roi_gray)

                color = (255, 0, 0)  # BGR 0-255
                stroke = 4
                end_cord_x = x + y + 20
                end_cord_y = y + h + 20
                cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

            #time.sleep(5)
            cv2.imshow('frame', frame)
            #time.sleep(5)
            if labels[id_] :#check for the lighting and camera
               namess = labels[id_]
               break

            if cv2.waitKey(20) & 0xff == ord('q'):
                break


        #messagebox.showinfo("welcome", namess)

        cap.release()
        cv2.destroyAllWindows()
        return namess
    def sound_alarm(path):
        playsound.playsound(path)

    def eyear(eye):
        A=dist.euclidean(eye[1],eye[5])#vertical sqrt((x2-x1)**2)
        B=dist.euclidean(eye[2],eye[4])#vertical
        C=dist.euclidean(eye[0],eye[3])#horizontal
        ear=(A+B)/(2.0*C)
        return ear
    def mouthar(mouth):
        h= dist.euclidean(mouth[0],mouth[4])
        v = 0
        for c in range(1,4):
            v += dist.euclidean(mouth[c],mouth[8-c])
            mar= v/(h*3)
            return mar



    def sleep():
        shape_predictor_path = os.path.join('shape_predictor_68_face_landmarks.dat')
        alarm = os.path.join('alarm.wav')

        ap = argparse.ArgumentParser()
        ap.add_argument("-w", "--webcam", type=int, default=0,
                        help="index of webcam on system")
        # this argument is used to locate any additional camera or usb camera
        args = vars(ap.parse_args())

        EYE_THRESH = 0.3  # if ear goes less than this frames counting starts
        EYE_AR_CONSEC_FRAMES = 48  # no of frames from eyes are closed
        COUNTER = 0
        ALARM_ON = False
        MOUTH_THRESH = 0.03
        MOUTH_AR_CONSEC_FRAMES = 10
        COUNTERR = 0

        print("Loading facial landmark")
        detector = dlib.get_frontal_face_detector()
        # print("done done")
        predictor = dlib.shape_predictor(shape_predictor_path)
        (lstart, lend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rstart, rend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        # print(lstart,lend)
        # print(rstart,rend)
        print("starting video stream thread...")
        vs = VideoStream(src=args["webcam"]).start()
        time.sleep(1.0)
        while True:
            frame = vs.read()
            frame = imutils.resize(frame, width=900)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                lefteye = shape[lstart:lend]
                righteye = shape[rstart:rend]
                leftear = eyear(lefteye)
                rightear = eyear(righteye)
                # print(lefteye)
                # print(righteye)
                # print(leftear)
                # print(rightear)
                inner_lips = shape[60:68]
                maar = mouthar(inner_lips)
                #print(maar)

                ear = (leftear + rightear) / 2.0

                lefteyehull = cv2.convexHull(lefteye)
                righteyehull = cv2.convexHull(righteye)
                cv2.drawContours(frame, [lefteyehull], -1, (255, 255, 0), 3)
                cv2.drawContours(frame, [righteyehull], -1, (255, 255, 0), 3)
                mouthHull = cv2.convexHull(inner_lips)
                cv2.drawContours(frame, [mouthHull], -1, (0, 0, 255), 5)
                if ear < EYE_THRESH:
                    COUNTER += 1
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        # print(COUNTER)
                        if not ALARM_ON:
                            ALARM_ON = True
                            if alarm != "":
                                t1 = Thread(target=sound_alarm(alarm))
                                t1.deamon = True
                                t1.start()

                        cv2.putText(frame, "open your eyse!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 225, 0), 2)
                else:
                    COUNTER = 0
                    ALARM_ON = False

                cv2.putText(frame, "EAR:{:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if maar > MOUTH_THRESH:
                    COUNTERR += 1
                    if COUNTERR >= MOUTH_AR_CONSEC_FRAMES:
                        #print(COUNTERR)
                        if not ALARM_ON:
                            ALARM_ON = True
                            if alarm != "":
                                t2 = Thread(target=sound_alarm(alarm))
                                t2.deamon = True
                                t2.start()

                        cv2.putText(frame, "Yaawning.....! take rest", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 225, 0),
                                    2)
                else:
                    COUNTERR = 0
                    ALARM_ON = False

                cv2.putText(frame, "MOUTH:{:.2f}".format(maar), (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(10) & 0xFF
            if key == ord("q"):
                screen.destroy()
                break
        cv2.destroyAllWindows()
        vs.stop()
        print("done now")

    screen = Tk()
    screen.geometry("300x250")
    screen.title("WELLCOME")
    screen.configure(background='turquoise')
    Button(screen, text="WellCome", height="2", width="20", fg='white', bg='purple', font=('comicsans', 12),
                command=login).place(x=70, y=100)
    Label(screen, text="RECONIZATION ", fg='purple', bg='turquoise', font=('comicsans', 10)).pack()
    screen.mainloop()
    #print("from tkinter")




main()
