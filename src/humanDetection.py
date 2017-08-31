import numpy as np
import imutils
import cv2
import time
import threading
import math

def alarm_info():
    print "Alert! target is not moving"

def alarm(switch):
    if switch == "on":
        timer.start()
    else: # switch == "off"
        timer.cancel()
        
def detectWantedPerson(name):

    # logic should be:
    # time_in_seconds = 5
    
    # if faceDetected within time_in_seconds:
    #     if personMatch(name):
    #         return True
    #     else:
    #         #other person
    #         return False
    # else:
    #     return False
    #     # time out
    
    return True

def personEntered():

    return True

def isPersonMovingWithin(time_in_seconds):

    start_time = time.time()

    # load the HOG Descriptor
    hog = cv2.HOGDescriptor()

    # low the default person detector
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
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
            # person detected
            draw_detections(frame,found)
            if isMoving(found):
                cv2.imshow('feed',frame)
            else:
                moving = False

            g_last_frame = found
            
        # check timeout
        if time.time() - start_time >= time_in_seconds:
            moving = True
            break

        if cv2.waitKey(1) == 27:
            #for debug purpose
            moving = True
            break
        
    cv2.destroyAllWindows()

def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

def isMoving(current_img):
    tolerance = 5 # 5 pixels tolerance
    for xc, yc, wc, hc in current_img:
        for xl, yl, wl, hl in g_last_frame:
            print "here 3"
            if math.fabs(xc - xl) > tolerance or math.fabs(yc - yl) > tolerance:
                return True
            else:
                return False
            

def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)





if __name__ == '__main__':

    detected = False
    person = "Lars"
    
    timer = threading.Timer(1.0, alarm_info)

    g_last_frame = ""
    g_timer_status = False
    
    if personEntered():

        # Now someone is entering the apartment
        if not detected:
            detected = detectWantedPerson(person)
            
        if detected:
            # We detected the correct person, keep tracking if he is moving
            while True:
                if not isPersonMovingWithin(5):
                    if not g_timer_status:
                        # No moving for a person, alarm!!
                        alarm("on")
                        g_timer_status = True
                else: # isPersonMovingWithin(5)
                    # target is moving
                    if g_timer_status:
                        alarm("off")
                        g_timer_status = False
#                if detectPersonleft():
#                    break

