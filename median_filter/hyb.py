import cv2

image = cv2.imread("ve.jpg")
processed_image = cv2.medianBlur(image, 3)
cv2.imshow('Median Filter Processing', processed_image)
cv2.imwrite('median.png', processed_image)
cv2.waitKey(0)
