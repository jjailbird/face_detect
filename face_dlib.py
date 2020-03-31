import numpy as np
import cv2
import dlib

faceCascade = cv2.CascadeClassifier('./haarcascade/haarcascade_frontalface_default.xml')
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

JAWLINE_POINTS = list(range(0, 17))
RIGHT_EYEBROW_POINTS = list(range(17, 22))
LEFT_EYEBROW_POINTS = list(range(22, 27))
NOSE_POINTS = list(range(27, 36))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
MOUTH_OUTLINE_POINTS = list(range(48, 61))
MOUTH_INNER_POINTS = list(range(61, 68))

def detectFace(frame):
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = faceCascade.detectMultiScale(gray, 1.05, 5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)

  for (x, y, w, h) in faces:
    dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

    landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, dlib_rect).parts()])
    landmarks_display = landmarks[0:68]

    for idx, point in enumerate(landmarks_display):
      pos = (point[0, 0], point[0, 1])
      cv2.circle(frame, pos, 2, color=(0, 255, 255), thickness=-1)

  return frame

capture = cv2.VideoCapture(0)

while(True):
  ret, frame = capture.read()

  detecedImage = detectFace(frame)
  cv2.imshow('face_detect', detecedImage)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

capture.release()
cv2.destroyAllWindows()  