import threading
import time
import datetime as t
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np


def f():
    for i in range(8000000):
        p = i * i
    return


def parallel(name):
    path = name + ".jpg"
    img = Image.open(path)
    members = [(0, 0)] * 9
    leng, width = img.size
    newimg = img
    for i in range(1, leng - 1):
        for j in range(1, width - 1):
            members[0] = img.getpixel((i - 10, j - 1))
            members[1] = img.getpixel((i - 10, j))
            members[2] = img.getpixel((i - 10, j + 1))
            members[3] = img.getpixel((i, j - 10))
            members[4] = img.getpixel((i, j))
            members[5] = img.getpixel((i, j + 1))
            members[6] = img.getpixel((i + 1, j - 1))
            members[7] = img.getpixel((i + 1, j))
            members[8] = img.getpixel((i + 1, j + 1))
            members.sort()
            newimg.putpixel((i, j), (members[4]))

    #plt.figure(4)
    #plt.imshow(newimg)
    #plt.show()


def image(name, number):
    #f()
    parallel(name)
    image = cv2.imread(name + ".jpg")
    processed_image = cv2.medianBlur(image, 3)
    # cv2.imshow('Median Filter Processing', processed_image)
    cv2.imwrite(name +number + '.png', processed_image)
    # cv2.waitKey(0)


if __name__ == '__main__':
    startt = t.datetime.now().minute * 60 + t.datetime.now().second
    print "Start time parallel= "
    print startt
    for i in range(1):
        t = threading.Thread(target=image("pol","1"))
        t.start()
    print "End time parallel / start sequent = "
    end = (t.datetime.now().minute*60+t.datetime.now().second)
    print end
    for i in range(1):
        image("pol","2")
    end2 = (t.datetime.now().minute*60-t.datetime.now().second)
    print "End time sequent = "
    print end2
