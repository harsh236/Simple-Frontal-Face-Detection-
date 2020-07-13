# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 18:55:18 2019

@author: im_harsh
"""


import cv2

face_cascade =cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade =cv2.CascadeClassifier('haarcascade_eye.xml')
 
def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for x,y,w,h in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
        roi_gray = gray[x:x+w,y:y+h]
        roi_frame = frame[x:x+w,y:y+h]
        eyes = eye_cascade.detectMultiScale(roi_gray,1.1,22)
        for ex,ey,ew,eh in eyes:
            cv2.rectangle(roi_frame,(ex,ey),(ex+w,ey+h),(0,255,0),3)
    return frame

video_capture = cv2.VideoCapture(0)
while(True):
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    response = detect(gray, frame)
    cv2.imshow('Video', response)
    if cv2.waitKey(1) & 0xFF == ord ('q'):
        break
    
video_capture.release()
cv2.destroyAllWindows()