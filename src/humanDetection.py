import numpy as np
import imutils
import cv2


def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)


if __name__ == '__main__':

    detected = False
    person = "Lars"
    
    while True:
        if detectPersonEntered():
            # Now someone is entering the apartment
            if not detected:
                detected = detectWantedPerson(person)

            # We detected the correct person
            if not isPersonMovingwithin(60):
                # No moving for a person, alarm!!
                alarm()
    


def detectWantedPerson(name):
    
    pattern = cv2.imread('100_75.jpg',0)
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

def detectPersonEntered():

    return True

def isPersonMovingWithin(time_in_seconds):

    # load the HOG Descriptor
    hog = cv2.HOGDescriptor()

    # low the default person detector
    hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )
    
    # open the video capture
    cap = cv2.VideoCapture(0)
    
    while True:
        _,frame=cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame = cv2.resize(frame, (640, 360), 0, 0, cv2.INTER_CUBIC)
        frame = cv2.resize(frame, (400, 225), 0, 0, cv2.INTER_CUBIC)
        
        # scale: which controls by how much the image is resized at each layer
        found,w=hog.detectMultiScale(frame, winStride=(4,4), padding=(32,32), scale=1.05)
        if not w is None:
            draw_detections(frame,found)
            cv2.imshow('feed',frame)
            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break
    cv2.destroyAllWindows()
