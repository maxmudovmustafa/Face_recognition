import os

from PIL import Image
from object_detection.utils import dataset_util

def create_tf_example(photo, person, iclass,
                      foldr):  # photo is the image name without extension, person is a folder name,
    # iclass is the index associated with the person, foldr: train or test folder
    # one image at a time
    img_f = os.path.join(foldr + person, photo + ".jpg")
    pic = Image.open(img_f)
    height = pic.height  # Image height
    width = pic.width  # Image width
    filename = str.encode(photo)  # Filename of the image. Empty if image is not from file
    image_data = tf.gfile.GFile(img_f, 'rb').read()  # encoded image data for tfrecord use

    image_format = b'jpeg'  # None #  or b'png'
    # declare coordinates
    xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = []  # List of normalized right x coordinates in bounding box
    # (1 per box)
    ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = []  # List of normalized bottom y coordinates in bounding box
    # (1 per box)
    classes_text = []  # List of string class name of bounding box (1 per box)
    classes = []  # List of integer class id of bounding box (1 per box)

    # our custom function to read labels from Labels/ directory
    txtlines = read_txt(person, photo)

    labels = txtlines.split()
    xmins.append(float(labels[1]) / width)  # divided by width for normalization
    xmaxs.append(float(labels[2]) / width)
    ymins.append(float(labels[3]) / height)
    ymaxs.append(float(labels[4]) / height)

    classes_text.append(str.encode(person))  # class name (person name)

    classes.append(iclass)  # class number associated with the person

    # the below code saves all the properties obtained above to tfrecord specific fields for object detection
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_example


def read_txt(person, photo):
    txtfile = labels + person + ".txt"  # labels should be a predetermined folder e.g. train or test
    txtfile_contents = open(txtfile, "r")
    txtlines = txtfile_contents.readlines()
    txtfile_contents.close()
    for line in txtlines:  # find a matching line that contains photo name and coordinates of the face
        if photo in line:
            txtlines = line
    return txtlines


import tensorflow as tf


def save_tf(folder):  # saving tfrecord
    tf_file = folder.split('/')[
                  -2] + '.tfrecord'  # folder names train and test, tfrecord names will also start with train and test
    writer = tf.python_io.TFRecordWriter('../working/' + tf_file)

    labelmap = '../working/' + 'object_label.pbtxt'  # for model training
    txtf = open(labelmap, "w")
    labels = '../working/' + 'labels.txt'  # for android deployment
    txtl = open(labels, "w")

    for ind, person in enumerate(os.listdir(folder)):
        iclass = ind + 1  # make sure label index starts from 1; zero is reserved
        txtf.write("item\n{\n  id: %s\n  name: '%s'\n}\n" % (iclass, person))  # for model training
        txtl.write("%s\n" % person)  # for android deployment
        for photo in os.listdir(folder + person):  # saving dataset and labels in tfrecord format by one image at a time
            tf_example = create_tf_example(photo.split('.')[0], person, iclass, folder)
            writer.write(tf_example.SerializeToString())
    txtf.close()
    writer.close()
