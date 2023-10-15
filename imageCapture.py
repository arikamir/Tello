from djitellopy import tello
import cv2

myTello = tello.Tello()

myTello.connect()
print(myTello.get_battery())

myTello.streamon()

while True:
    img = myTello.get_frame_read().frame
    img = cv2.resize(img,(360,240))
    cv2.imshow("Image", img)
    cv2.waitKey(1)
