"""Gute Suche vielleicht (aber noch nicht angeschaut)

face detection bounding box face masks

- für mtcnn muss erst tensorflow installiert werden, vielleicht 
funktioniert es dann, habe aber kein Internet. 
"""


import cv2
import dlib
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
#from mtcnn.mtcnn import MTCNN


def face_detection(frame, algorithm):
    if algorithm == "cascade":
        """Source: https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81
        
        viele verschiedene cascade möglichkeiten, sollte man durchlaufen lassen.
        https://github.com/anaustinbeing/haar-cascade-files/blob/master/haarcascade_frontalface_alt.xml
        """
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml') # 'haarcascade_frontalface_default.xml'

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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
    elif algorithm == "new":
        faces, _ = cv2.detect_face(frame)# loop through detected faces and add bounding box
        for face in faces: 
            bbox.append([face[0],face[1],face[2],face[3]])
    elif algorithm == "mediapipe":
        
        # STEP 2: Create an FaceDetector object.
        base_options = python.BaseOptions(model_asset_path='detector.tflite')
        options = vision.FaceDetectorOptions(base_options=base_options)
        detector = vision.FaceDetector.create_from_options(options)

        # STEP 4: Detect faces in the input image.
        detection_result = detector.detect(frame)
        bbox.append(detection_result.detection.bounding_box)
    
    # elif algorithm == "mtcnn":
    #     # https://machinelearningmastery.com/how-to-perform-face-detection-with-classical-and-deep-learning-methods-in-python-with-keras/
    #     detector = MTCNN()
    #     faces = detector.detect_faces(frame)
    #     bbox.append(faces.result["box"])

    else: 
        raise Exception("Dieser Algoritmus existiert nicht...")
    return bbox, frame
