import numpy as np
import imutils
import cv2
import time
from threading import Thread
from threading import Event
import math

class MyThread(Thread):

    period = 0
    isMoving = True
    pulse = 0
    
    def __init__(self, event):
        Thread.__init__(self)
        self.stopped = event

    def update(isMoving, pulse, not_moving_period):
        self.isMoving = isMoving
        self.pulse = pulse
        self.period = not_moving_period
        
    def run(self):
        while not self.stopped.wait(2):
            print "Target status: isMoving %s pulse %s period %s" \
                % (str(self.isMoving), str(self.pulse), str(self.period))

def status_on():
    stopFlag = Event()
    thread = MyThread(stopFlag)
    thread.start()

    return thread, stopFlag

def status_off(stopEvent):
    stopEvent.set()

def status_update(thread, isMoving, pulse, not_moving_period):
    thread.update(isMoving, pulse, not_moving_period)
        
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

    capture = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    success, frame = capture.read()
    counter = 50
    ret = False
    
    while success:
        print "counter: " + str(counter)
        
        success, frame = capture.read()
        res = cv2.resize(frame, (320, 200), 0, 0, cv2.INTER_CUBIC)
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 3)
        
        if len(faces):
            # face detected
            print "face detected!"
            print "faces: %s" %(faces,)
            for (x, y, w, h) in faces:
                cv2.rectangle(gray, (x, y), (x+w, y+h),(255,255,255),2)
                ret = True
            break
        
        #cv2.imshow('Gray', gray)
        #font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(background,'Pulse 44 bpm',(330,40), font, 1,(233,244,255),1,cv2.LINE_AA)

        rows, cols, channels = res.shape
        background[0:rows, 0:cols] = res
        cv2.imshow('Health monitor', background)
        k = cv2.waitKey(60) & 0xff
        if k == 27:
            ret = False
            break
        
        counter -= 1
        if (counter == 0):
            ret = False
            break
        
    #cv2.destroyWindow('Gray')
    capture.release()

    print "ret: " + str(ret)
    
    return ret

def detectPersonStatusWithin_haar(time_in_seconds):

    start_time = time.time()
    last_frame = ""
    moving = "False"
    pulse = 60
    
    # open the video capture
    cap = cv2.VideoCapture(0)

    # load the HOG Descriptor
    hog = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    while True:
        # This loop will last only for time_in_seconds seconds
        
        _,frame=cap.read()
        frame = cv2.resize(frame, (320, 200), 0, 0, cv2.INTER_CUBIC)
        health_f = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # TODO: check resolution
        
        faces = face_cascade.detectMultiScale(gray, 1.1, 3)
        
        if len(faces):
            # person detected
            draw_detections(frame,faces)
            rows, cols, channels = frame.shape
            background[0:rows, 0:cols] = frame
            cv2.imshow('Health monitor', background)

            if isMoving(faces, last_frame):
                moving = True
                print "target is moving! detection done!"
                break
            
            last_frame = faces
        else:
            print "no person detected!!"
            
        # check timeout
        if time.time() - start_time >= time_in_seconds:
            print "time out. target not moved within %s seconds" % (time_in_seconds)  
            break

    cap.release()

    return moving, pulse

def detectPersonStatusWithin(time_in_seconds):

    start_time = time.time()
    last_frame = ""
    moving = "False"
    pulse = 60
    
    # load the HOG Descriptor
    hog = cv2.HOGDescriptor()

    # low the default person detector
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    # open the video capture
    cap = cv2.VideoCapture(0)
    counter = 0
    while True:
        # This loop will last only for time_in_seconds seconds
        
        _,frame=cap.read()
        frame = cv2.resize(frame, (320, 200), 0, 0, cv2.INTER_CUBIC)
        health_f = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # TODO: check resolution
        
        # scale: which controls by how much the image is resized at each layer
        found,w = hog.detectMultiScale(health_f, winStride=(4,4), padding=(32,32), scale=1.2)
        
        if not w is None:
            # person detected
            draw_detections(frame,found)
            rows, cols, channels = frame.shape
            background[0:rows, 0:cols] = frame
            cv2.imshow('Health monitor', background)

            # if isMoving(found, last_frame):
            #     moving = True
            #     print "target is moving! detection done!"
            #     break
            
            last_frame = found
        else:
            print "no person detected!!"
        counter += 1
        if counter == 50:
            counter = 0
            # update status on the frame
            
        # check timeout
        if time.time() - start_time >= time_in_seconds:
            print "time out. target not moved within %s seconds" % (time_in_seconds)  
            break

    cap.release()

    return moving, pulse

def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

def isMoving(current_img, last_img):
    tolerance = 5 # 5 pixels tolerance
    for xc, yc, wc, hc in current_img:
        for xl, yl, wl, hl in last_img:
            if math.fabs(xc - xl) > tolerance or math.fabs(yc - yl) > tolerance:
                print "is moving"
                return True
            else:
                print "is NOT moving"
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
    
    background = cv2.imread('../data/background.jpg')
    cv2.imshow('Health monitor', background)

    g_timer_status = False
    g_timer_thread = ""
    g_timer_stop_event = ""
    
    while True:
        if not detected:
            # Now someone is entering the apartment
            detected = detectWantedPerson(person)

        if detected:
            # start a thread that keep updating the target status
            # g_timer_thread, g_timer_stop_event = status_on()

            _, _ = detectPersonStatusWithin(2)
            try:
                # We detected the correct person, keep tracking if he is moving
                while True:
                    isMoving, Pulse = detectPersonStatusWithin(300)
                    #status_update(g_timer_thread, isMoving, Pulse, 30)
                    print "sleeping!!"
                    time.sleep(5)
                    print "Waking UP!!"
                    if cv2.waitKey(1) == 27:
                        break

            except (RuntimeError, TypeError, NameError):
                #status_off(g_timer_stop_event)
                pass
            
            #status_off(g_timer_stop_event)            
