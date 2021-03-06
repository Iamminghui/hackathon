import cv2
import numpy as np

#pattern = cv2.imread('32_32.jpg',0)
pattern = cv2.imread('../data/100_75.jpg',0)
orb = cv2.ORB_create()

capture = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
success, frame = capture.read()

kp1, des1 = orb.detectAndCompute(pattern, None)
kp2, des2 = orb.detectAndCompute(frame, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda X:X.distance)

img3 = cv2.drawMatches(pattern, kp1, frame, kp2, matches[:30], None, flags=2)

while success:
    success, frame = capture.read()
    res = cv2.resize(frame, (320, 200), 0, 0, cv2.INTER_CUBIC)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 3)
    kp1, des1 = orb.detectAndCompute(pattern, None)
    kp2, des2 = orb.detectAndCompute(gray, None)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda X:X.distance)
    for (x, y, w, h) in faces:
        cv2.rectangle(gray, (x, y), (x+w, y+h),(255,255,255),2)
    gray = cv2.drawMatches(pattern, kp1, gray, kp2, matches[:10], None, flags=2)
    cv2.imshow('Gray', gray)

    k = cv2.waitKey(60) & 0xff
    if k == 27:
        break
cv2.destroyAllWindows()
capture.release()
