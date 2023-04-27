import cv2
import dlib

def face_detection(frame, algorithm):
    if algorithm == "cascade":
        """Source: https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81"""
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

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
    return bbox, frame
