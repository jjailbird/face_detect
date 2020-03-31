import numpy as np
import cv2

faceCascade = cv2.CascadeClassifier('./haarcascade/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade/haarcascade_eye.xml')

def detect(frame):
  # detectMultiScale(const Mat& image, vector<Rect>& objects, double scaleFactor=1.1, int minNeighbors=3, int flags=0, Size minSize=Size(), Size maxSize=Size())  
  
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = faceCascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)

  for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    face_gray = gray[y: y+h, x: x+w]
    face_color = frame[y: y+h, x: x+w]

    eyes = eyeCascade.detectMultiScale(face_gray, 1.1, 3)

    for (ex, ey, ew, eh) in eyes:
      cv2.rectangle(face_color, (ex, ey), (ex + ew, ey + eh),(0,255,0),  1)

  return frame

video = cv2.VideoCapture(0)

while(True):
  ret, frame = video.read()
  
  faceDetectedImage = detect(frame)
  cv2.imshow('result', faceDetectedImage)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

video.release()
cv2.destroyAllWindows()
