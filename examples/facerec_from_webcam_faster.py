import glob
import subprocess
import time

import cv2
import psutil
import pyttsx3
import self as self
from PIL import Image, ImageDraw

import face_recognition

# video_capture = cv2.VideoCapture("rtsp://admin:Aa6369750@192.168.0.11:555/Streaming/channels/2/")
video_capture = cv2.VideoCapture(0)



def speak(text):
    engine = pyttsx3.init()
    engine.say(text=text)
    engine.runAndWait()


first_person = face_recognition.load_image_file("wh.jpg")
first_face_encoding = face_recognition.face_encodings(first_person)[0]

second_image = face_recognition.load_image_file("Shoxruxbek.jpg")
second_face_encoding = face_recognition.face_encodings(second_image)[0]

known_face_encodings = [
    first_face_encoding,
    second_face_encoding
]

known_face_names = [
    "SHoxruxbek",
    "Maxmudov Shoxruxbek",
    "Ubaydullayev Johongir",
]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


def getSizedFrame(width, height):
    """Function to return an image with the size I want"""
    s, img = self.cam.read()

    # Only process valid image frames
    if s:
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return s, img


o = False
while True:
    ret, frame = video_capture.read()
    time.sleep(0.5)
    cv2.imwrite("framed.jpg", frame)
    find_person = face_recognition.load_image_file('framed.jpg')
    faceLoc = face_recognition.face_locations(find_person)
    faceEnc = face_recognition.face_encodings(find_person, faceLoc)
    pil_image = Image.fromarray(find_person)
    draw = ImageDraw.Draw(pil_image)
    if o:
        for proc in psutil.process_iter():
            if proc.name() == "display":
                proc.kill()
    for (top, right, bottom, left), face_encoding in zip(faceLoc, faceEnc):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown Person"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(0, 0, 0))
    del draw
    o = True
    pil_image.show()
    if 0xFF == ord('q'): break

    # cv2.imshow('Video', draw)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
