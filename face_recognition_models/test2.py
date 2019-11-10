import pickle

import cv2

face_cascade = cv2.CascadeClassifier(
    "/home/warlock/Downloads/pych/face_recognition/data/haarcascade_frontalface_alt2.xml")
face__casade_right = cv2.CascadeClassifier("/home/warlock/Downloads/pych/face_recognition/data/haarcascade_right.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("/home/warlock/Downloads/pych/face_recognition/face_recognition_models/recognizers/trainner.yml")
labels = {"person_name": 1}
with open("/home/warlock/Downloads/pych/face_recognition/face_recognition_models/pickles/labels.pickles", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

video_capture = cv2.VideoCapture("rtsp://admin:Aa6369750@192.168.1.100:555/Streaming/channels/2/", cv2.CAP_FFMPEG)
found_face = False
while True:
    red, frame = video_capture.read()
    grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(grey, 1.2, 5)
    # faces_right = face__casade_right.detectMultiScale(grey, 1.5, 5)
    # print faces_right
    for (x, y, w, h) in faces:
        roi_grey = grey[y: y + h, x:x + w]
        # cv2.imwrite("my_image.png")
        # print x
        id_, conf = recognizer.predict(roi_grey)
        print conf
        if 70 <= conf < 100:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 100, 100)
            stroke = 2
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)
            found_face = True
        end_cord_x = x + w
        end_cord_y = y + h
        if found_face:
            cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), (0, 255, 0), 2)
            # cv2.imwrite("name.jpg", frame)
            cv2.imwrite(name + ".jpg", frame)
        else:
            cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), (0, 0, 255), 2)
        found_face = False
    cv2.imshow("vidoe", frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# if found_face:
#         engine = pyttsx3.init()
#         engine.say("I Found  "+name+" In the picture")
#         engine.runAndWait()

video_capture.release()
cv2.destroyAllWindows()
