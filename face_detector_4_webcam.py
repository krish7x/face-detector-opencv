import cv2
from random import randrange

# importing the trained XML data
tranined_frontalface_data = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml")

# using webcam
webcam = cv2.VideoCapture(0)

# looping through every single frame
while True:
    # read the webcam
    successful_frame_read, frame = webcam.read()

    # conversion to gray scale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # collecting the face co-ordinates
    face_coordinates = tranined_frontalface_data.detectMultiScale(
        grayscaled_img)

    #face_coordinates = [[x,y,w,h]]
    # allocating the first element in the list
    # (x, y, w, h) = face_coordinates[]

    # looping through multiple faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      (randrange(256), randrange(256), randrange(256)), 2)

    # show the output
    cv2.imshow("My webCam", frame)
    key = cv2.waitKey(1)

    # if the escape or 'q'key is pressed then it will force quit
    if key == 27 or key == 113:
        break

# webcam will get closed
webcam.release()
