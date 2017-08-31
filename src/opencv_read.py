import numpy as np
import cv2

cap = cv2.VideoCapture(0)

out = cv2.VideoWriter(
              'output1.avi',
              cv2.VideoWriter_fourcc(*'MJPG'),
              20.0,
              (640,480))
x = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        #frame = cv2.Color(frame, cv2.COLOR_BRG2GRAY)

        # write the flipped frame
        out.write(frame)

        #cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
    x = x + 1
    if x >= 200:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
