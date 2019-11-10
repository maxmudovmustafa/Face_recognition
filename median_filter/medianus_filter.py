from PIL import Image
import matplotlib.pyplot as plt

path = "ve.jpg"
img = Image.open(path)
members = [(0, 0)] * 9
leng, width = img.size
# newimg = Image.new("RGB", (leng, width), "white")
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

# img=mpimg.imread('V.jpg')

plt.figure(4)
imgplot = plt.imshow(newimg)
plt.show()
