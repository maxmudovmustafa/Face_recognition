def face_dnn(img, coord=False):
    blob = cv2.dnn.blobFromImage(img, 1, (224, 224), [104, 117, 123], False, False)  #
    # params: source, scale=1, size=300,300, mean RGB values (r,g,b), rgb swapping=false, crop = false
    conf_threshold = 0.8  # confidence at least 80%
    frameWidth = img.shape[1]  # get image width
    frameHeight = img.shape[0]  # get image height
    max_confidence = 0
    net.setInput(blob)
    detections = net.forward()
    detection_index = 0
    bboxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            if max_confidence < confidence:  # only show maximum confidence face
                max_confidence = confidence
                detection_index = i
    i = detection_index  # face location with maximum confidence
    x1 = int(detections[0, 0, i, 3] * frameWidth)  # each of i corresponds to xmin, ymin, xmax and ymax
    y1 = int(detections[0, 0, i, 4] * frameHeight)
    x2 = int(detections[0, 0, i, 5] * frameWidth)
    y2 = int(detections[0, 0, i, 6] * frameHeight)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)  # draw a rectangle on a detected area
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # save image
    if coord == True:
        return x1, y1, x2, y2  # returns coordinates only

    return cv_rgb  # returns annotated image



# import os
#
# import tensorflow as tf
#
#
# def save_tf(folder):  # saving tfrecord
#     tf_file = folder.split('/')[
#                   -2] + '.tfrecord'  # folder names train and test, tfrecord names will also start with train and test
#     writer = tf.python_io.TFRecordWriter('../working/' + tf_file)
#
#     labelmap = '../working/' + 'object_label.pbtxt'  # for model training
#     txtf = open(labelmap, "w")
#     labels = '../working/' + 'labels.txt'  # for android deployment
#     txtl = open(labels, "w")
#
#     for ind, person in enumerate(os.listdir(folder)):
#         iclass = ind + 1  # make sure label index starts from 1; zero is reserved
#         txtf.write("item\n{\n  id: %s\n  name: '%s'\n}\n" % (iclass, person))  # for model training
#         txtl.write("%s\n" % person)  # for android deployment
#         for photo in os.listdir(folder + person):  # saving dataset and labels in tfrecord format by one image at a time
#             tf_example = create_tf_example(photo.split('.')[0], person, iclass, folder)
#             writer.write(tf_example.SerializeToString())
#     txtf.close()
#
# writer.close()
