import face_recognition
import cv2
import numpy as np
import face_recognition

import os
# from object_detection.utils import dataset_utils

video_capture = cv2.VideoCapture(0)
# video_capture = open_cam_rtsp("rtsp://192.168.0.4/live1.sdp", 1280, 720, 200)
def open_cam_rtsp(uri, width, height, latency):
    gst_str = ('rtspsrc location={} latency={} ! '
               'rtph264depay ! h264parse ! omxh264dec ! '
               'nvvidconv ! '
               'video/x-raw, width=(int){}, height=(int){}, format=(string)BGRx ! '
               'videoconvert ! appsink').format(uri, latency, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


# video_capture = cv2.VideoCapture("dron.mp4", cv2.IMREAD_GRAYSCALE)

instances = []

# Load in the images
for filepath in os.listdir('knn_examples/test/'):
    instances.append(cv2.imread('knn_examples/test   /{0}'.format(filepath),0))

print(type(instances[0]))

first_person = face_recognition.load_image_file("test.png")
first_face_encoding = face_recognition.face_encodings(first_person)[0]

second_image = face_recognition.load_image_file("Shoxruxbek.jpg")
second_face_encoding = face_recognition.face_encodings(second_image)[0]

known_face_encodings = [
    first_face_encoding,
    second_face_encoding
]

known_face_names = [
    "Ubaydullayev Johongir",
    "Maxmudov Shoxruxbek"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 220), 2)

        # cv2.rectangle(frame, (left-35, bottom - 55), (right -35 , bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left - 46, bottom + 26), font, 1.0, (0, 0, 220), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
