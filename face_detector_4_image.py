import cv2
from random import randrange

# importing the trained XML data
tranined_frontalface_data = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml")


# read Image
img = cv2.imread("./images/my3.jpg")

# conversion to gray scale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect faces multi-scale --> whatever the size(scale) is just detect the image
face_coordinates = tranined_frontalface_data.detectMultiScale(grayscaled_img)

# printing face coordinates
print(face_coordinates)

#face_coordinates = [[x,y,w,h]]

# allocating the first element in the list
# (x, y, w, h) = face_coordinates[]

# drawing rectangle
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x + w, y + h),
                  (randrange(190, 256), randrange(190, 256), randrange(190, 256)), 2)

'''
1 -> img 
2 -> top left 
3 -> bottom right(plus width, plus height)
4 -> color of the rectangle , 5 -> thickness of the rectangle
'''

# show image
cv2.imshow("Face Detector", img)

# closes after the key is pressed
cv2.waitKey()
