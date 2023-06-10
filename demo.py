import cv2
import argparse, os
import torch

from torchvision import transforms
import numpy as np
from model import model_static
from PIL import Image, ImageDraw, ImageFont
from colour import Color

# self written function for the face detectors
import face_detection_functions


# für Laufzeitanalysen
import cProfile
from pstats import Stats



parser = argparse.ArgumentParser()

parser.add_argument('--video', type=str, help='input video path. live cam is used when not specified')
parser.add_argument('--model_weight', type=str, help='path to model weights file', default='data/model_weights.pkl')
parser.add_argument('-save_vis', help='saves output as video', action='store_true')
parser.add_argument('-save_text', help='saves output as text', action='store_true')
parser.add_argument('-display_off', help='do not display frames', action='store_true')
parser.add_argument('--face_detector', type=str, help='the face detector from face_detection_functions.py', default="cascade")

args = parser.parse_args()


def drawrect(drawcontext, xy, outline=None, width=0):
    """Draws the rectangle around the face in the final result.

    Args:
        drawcontext (function): The plotting function
        xy (list): contains x,y values for the top left corner and for the lower right corner
        outline (str, optional): The color for the outline of the rectangle in hex. Defaults to None.
        width (int, optional): The broadness of the rectangle. Defaults to 0.
    """
    (x1, y1), (x2, y2) = xy
    points = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
    drawcontext.line(points, fill=outline, width=width)


def run(video_path, model_weight, vis, display_off, save_text, face_detector):
    # set up vis settings
    
    # das hier wird so invertiert, dass dort später wieder rot draus wird
    red = Color("blue")
    colors = list(red.range_to(Color("green"),10))
    font = ImageFont.truetype("data/arial.ttf", 40)
    
    # set up video source
    if video_path is None:
        cap = cv2.VideoCapture(0)
        video_path = 'live.avi'
    else:
        cap = cv2.VideoCapture(video_path)

    # set up output file
    if save_text:
        outtext_name = os.path.basename(video_path).replace('.mp4','_output.txt')
        f = open(outtext_name, "w")
    if vis:
        outvis_name = os.path.basename(video_path).replace('.avi','_output.avi')
        imwidth = int(cap.get(3)); imheight = int(cap.get(4))
        outvid = cv2.VideoWriter(outvis_name,cv2.VideoWriter_fourcc('M','J','P','G'), cap.get(5), (imwidth,imheight))

    if (cap.isOpened()== False):
        print("Error opening video stream or file")
        exit()

    frame_cnt = 0

    # set up data transformation
    test_transforms = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # load model weights
    model = model_static(model_weight)
    model_dict = model.state_dict()
    snapshot = torch.load(model_weight)
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)

    model.cuda()
    #model.to(torch.device('cuda:1'))
    model.train(False)

    # video reading loop
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame_cnt+=1

            # calls a face detector function to return the rectangular coordinates around the detected face.  
            bbox = face_detection_functions.face_detection(frame, face_detector)

            frame = Image.fromarray(frame, mode = "RGB")
            for b in bbox:

                # because the model is only trained for the face and not the whole frame, 
                # the face is cropped to the detected bounding box around the face 
                face = frame.crop((b))
                img = test_transforms(face)
                img.unsqueeze_(0)

                # forward pass; evaluation of the face with the model
                output = model(img.cuda())
                score = torch.sigmoid(output).item()
                score = float(score)
                coloridx = min(int(round(score)*10),9)

                # draw video frame with the original frame, a bounding box around the face and the rounded calculated score 
                draw = ImageDraw.Draw(frame)
                drawrect(draw, [(b[0], b[1]), (b[2], b[3])], outline=colors[coloridx].hex, width=5)
                draw.text((b[0],b[3]), str(round(score,2)), fill=(255,255,255,128), font=font)
                if save_text:
                    f.write("%d,%f\n"%(frame_cnt,score))
                    if frame_cnt >= 20500:
                        print("DONE!")
                        f.close()
                        raise KeyError


            if not display_off:
                # convert PIL image back to opencv format for faster display
                frame = np.asarray(frame)
                cv2.imshow('',frame)
                if vis:
                    outvid.write(frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        else:
            break

    if vis:
        outvid.release()
    if save_text:
        f.close()
    cap.release()
    print('DONE!')

if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()
    run(args.video, args.model_weight, args.save_vis, args.display_off, args.save_text, args.face_detector)
    pr.disable()
    stats = Stats(pr)
    stats.sort_stats('tottime').print_stats(10)