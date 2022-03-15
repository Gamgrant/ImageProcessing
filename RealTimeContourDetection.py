import numpy as np
import cv2
import functools
import operator

cap = cv2.VideoCapture(1)
# Array of 2 by 2 containing, the unsigned ingteger; need for morphological transformations
kernel = np.ones((2, 2), np.uint8)

while (True):

    # set the resolution of the camera to 1920 by 1080
    cap.set(3, 800)
    cap.set(4, 450)

    # capture frame-by-frame
    ret, frame = cap.read()

    # Our operations o the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    gray = cv2.medianBlur(gray, 3)  # to remove the salt and paper noise
    # to binary
    # to detect the white objects
    ret, thresh = cv2.threshold(gray, 200, 255, 0)
    # to get the outer boundary only
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)
    # to strengthen the weak pixels
    thresh = cv2.dilate(thresh, kernel, iterations=5)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        cv2.drawContours(frame, contours, - 1, (0, 255, 0), 5)
        # apply the bounding rectangle around all the objects
        for cntr in contours:
            x, y, w, h = cv2.boundingRect(cntr)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            extBot = tuple(cntr[cntr[:, :, 1].argmax()][0])
            botdots = cv2.circle(frame, extBot, 8, (255, 255, 0), -1)
            tupletoStr = ','.join(map(str, extBot))
            print(tupletoStr)
            cv2.putText(frame, tupletoStr, extBot,
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2,)
            # changes made here
         # print("x,y,w,h:", x, y, w, h)
         # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
