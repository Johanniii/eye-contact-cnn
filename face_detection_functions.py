import cv2
def cascade(frame):
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
    return bbox, frame
