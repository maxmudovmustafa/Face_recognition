import cv2
import numpy as np
import threading
import time


class myThread(threading.Thread):
    def __init__(self, threadID, name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name

    def run(self):
        image(self.name)


def image(name):
    image = cv2.imread(name + ".jpg")
    processed_image = cv2.medianBlur(image, 3)
    cv2.imshow('Median Filter Processing', processed_image)
    cv2.imwrite('median.png', processed_image)
    print time.ctime(time.time())
    cv2.waitKey(0)


thr1 = myThread(1, "ve")

print time.ctime(time.time())

thr1.start()

thr1.join()

# processed_image = cv2.medianBlur(image, 3)
