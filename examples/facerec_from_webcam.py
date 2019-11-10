import cv2

# video_capture = cv2.VideoCapture(0)
import numpy as np

import face_recognition
video_capture = cv2.VideoCapture(0)
# video_capture = cv2.VideoCapture("rtsp://admin:Aa6369750@192.168.1.100:555/Streaming/channels/2/", cv2.CAP_FFMPEG)
if not video_capture.isOpened():
    print('VideoCapture not opened')
    exit(-1)


my_image = face_recognition.load_image_file("Shoxruxbek.jpg")
my_image_encoding = face_recognition.face_encodings(my_image)[0]

known_face_encodings = [
    my_image_encoding
]
known_face_names = [
    "Shoxrux"
]
name = "Unknown"

while True:
    ret, frame = video_capture.read()

    face_locations = face_recognition.face_locations(frame)
    if face_locations:
        # break
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
