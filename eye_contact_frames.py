from bs4 import BeautifulSoup
import numpy as np


def eye_contact_frames(file, total_frames):
    
    with open(file, 'r') as f:
        data = f.read()
    data = BeautifulSoup(data, 'xml')
    box_list = data.find_all('box')
    eye_contact_frames = [int(box['frame']) for box in box_list]

    eye_contact = np.zeros(total_frames)
    np.put(eye_contact, eye_contact_frames, 1)

    return eye_contact