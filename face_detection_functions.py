"""Gute Suche vielleicht (aber noch nicht angeschaut)

face detection bounding box face masks
"""


import cv2
import dlib
from mediapipe import tasks
import mediapipe as mp

# dafür erst tensorflow installieren
from mtcnn.mtcnn import MTCNN
import numpy as np

# version 0.9.3.0 - niedrigere funktionieren nicht, aber jetzt ist ein anderer Fehler
from mediapipe import tasks
from mediapipe.tasks import python

def face_detection(frame, algorithm):
    if algorithm == "cascade":
        """Source: https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81
        
        viele verschiedene cascade möglichkeiten, sollte man durchlaufen lassen.
        https://github.com/anaustinbeing/haar-cascade-files/blob/master/haarcascade_frontalface_alt.xml
        """
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml') # 'haarcascade_frontalface_default.xml'

        bbox = []

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect the faces (mehr als nötig?)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # adds the coordinates of the face Umriss to bbox
        for (x,y,w,h) in (faces):
            bbox.append([x,y,x+w,y+h])

    elif algorithm == "dlib":
        CNN_FACE_MODEL = 'data/mmod_human_face_detector.dat' # from http://dlib.net/files/mmod_human_face_detector.dat.bz2
        cnn_face_detector = dlib.cnn_face_detection_model_v1(CNN_FACE_MODEL)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bbox = []
        dets = cnn_face_detector(frame, 1)
        for d in dets:
            l = d.rect.left()
            r = d.rect.right()
            t = d.rect.top()
            b = d.rect.bottom()
            # expand a bit
            l -= (r-l)*0.2
            r += (r-l)*0.2
            t -= (b-t)*0.2
            b += (b-t)*0.2
            bbox.append([l,t,r,b])

    elif algorithm == "mediapipe":
        """ Für diesen algorithmus braucht man vermutlich eher zugeschnittene Videos, es scheint für Selfie-Sicht optimiert zu sein... 
        Sonst erkennt der aber sogar Ärzte, die mit cascade fast gar nicht erkannt werden können
        """

        bbox = []
        # https://github.com/googlesamples/mediapipe/blob/main/examples/face_detector/python/face_detector.ipynb
        # STEP 2: Create an FaceDetector object.
        base_options = python.BaseOptions(model_asset_path='face_detection_short_range.tflite')
        options = python.vision.FaceDetectorOptions(base_options=base_options)
        detector = python.vision.FaceDetector.create_from_options(options)

        frame = mp.Image(image_format= mp.ImageFormat.SRGB, data = frame)
        
        # STEP 4: Detect faces in the input image.
        detection_result = detector.detect(frame)
        if detection_result.detections:
            bounding_box = detection_result.detections[0].bounding_box
            bbox.append([bounding_box.origin_x, bounding_box.origin_y, bounding_box.width + bounding_box.origin_x, bounding_box.height + bounding_box.origin_y])

    elif algorithm == "mtcnn":
        """ Nicht besonders schnell, vermutlich nicht zielführend """
        # https://machinelearningmastery.com/how-to-perform-face-detection-with-classical-and-deep-learning-methods-in-python-with-keras/
        detector = MTCNN()
        faces = detector.detect_faces(frame)
        bbox = []
        print(faces)
        if faces : 
            results = faces[0]["box"]
            bbox  = [[results[0], results[1], results[0]+ results[2], results[1]+ results[3]]]
    
    else: 
        raise Exception("Dieser Algoritmus existiert nicht...")
    return bbox