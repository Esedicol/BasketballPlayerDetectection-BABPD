

# Author Emmanuel Sedicol
from collections import deque
from imutils.video import VideoStream
from PIL import Image  
import PIL  
import numpy as np
import argparse
import cv2
import imutils
import time
import os
import matplotlib.pyplot as plt
import tensorflow as tf
counter = 1

xx = []
yy = []
CATEGORIES = ['basketball', 'hoop']

model = tf.keras.models.load_model("/Users/esedicol/Desktop/Desktop/FYP/Ball_Tracking/POSE_ESTIMATION/CNN_MODEL")

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
    help="path to the (optional) video file")
args = vars(ap.parse_args())

def prepare_ball(f):
    image = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    pred_img = cv2.resize(image, (80,80))
    final_image = pred_img.reshape(-1, 80, 80, 1)
    prediction = model.predict([final_image])
    final_pred = CATEGORIES[int(prediction[0][0])]
    return(final_pred)

orangeLower = (0, 80, 110)
orangeUpper = (8,200,175)

vs = cv2.VideoCapture(args["video"])

# allow the camera or video file to warm up
time.sleep(2.0)

# keep looping
while True:
    # grab the current frame
    _,frame = vs.read()

    if frame is None:
        break

    frame = cv2.resize(frame,(600, 400))
    frame2 = frame.copy()

    blur = cv2.GaussianBlur(frame, (15, 15),0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, orangeLower, orangeUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # only proceed if at least one contour was found
    if len(cnts) > 0:

        # Retrieve the largest contour in the mask, then use it to compute the minimum enclosing circle and centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        # Using moments theory to calculate centroid 
        M = cv2.moments(c)
        x_coor = int(M["m10"] / M["m00"])
        y_coor = int(M["m01"] / M["m00"])

        frame2 = frame[int(y - 20):int(y + 20),int(x - 20):int(x + 20)] 
        height, width, _ = frame.shape
        height2, width2, _ = frame2.shape


        print(f'1: {width} - {height}')
        print(f'2: {width2} - {height2}')

        if height2 < height and width2 < width:
            if prepare_ball(frame2) == 'basketball':
            # Append (x,y) coordinates and plot them
                xx.append(x)
                yy.append(y)

                # Draw the circle and centroid on the frame,
                cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
                cv2.circle(frame, (int(x), int(y)), 1, (0, 0, 255), 2)

                # Draw rectangle around the minimum enclosed circle
                cv2.rectangle(frame, (int(x - 20), int(y - 20)), (int(x + 20), int(y + 20)), (255,0,0), 2)
                cv2.putText(frame, "Basketball", (int(x - 30), int(y - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1 )
        else:
            cv2.imshow('frame',frame)




    cv2.imshow('frame',frame)
    # cv2.imshow('Mask', mask)

    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break


# Plot (x,y) coordinates of basketball
plt.scatter(xx, yy)
plt.gca().invert_yaxis()
plt.show()


vs.release()
# close all windows
cv2.destroyAllWindows()

arguments = {"keywords":"Basketball net","limit":20,"print_urls":True}es
res = google_images_download.googleimagesdownload()









