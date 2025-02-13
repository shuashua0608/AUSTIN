import argparse
import imutils
import time
import dlib
import cv2
import os
import warnings


warnings.filterwarnings('ignore')

import math

import matplotlib.pyplot as plt
from torchvision import transforms

import torch

from vidstab.VidStab import VidStab
from PIL import Image, ImageDraw, ImageFilter, ImageChops
from imutils.video import FileVideoStream
from imutils import face_utils
from eco import ECOTracker
import numpy as np
from PIL import Image
from dectect import AntiSpoofPredict

from pfld.pfld import PFLDInference, AuxiliaryNet
warnings.filterwarnings('ignore')

device = "cuda"

def get_num(point_dict,name,axis):
    num = point_dict.get(f'{name}')[axis]
    num = float(num)
    return num

def cross_point(line1, line2):  
    x1 = line1[0]  
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]

    x3 = line2[0]
    y3 = line2[1]
    x4 = line2[2]
    y4 = line2[3]

    k1 = (y2 - y1) * 1.0 / (x2 - x1) 
    b1 = y1 * 1.0 - x1 * k1 * 1.0  
    if (x4 - x3) == 0: 
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)
        b2 = y3 * 1.0 - x3 * k2 * 1.0
    if k2 == None:
        x = x3
    else:
        x = (b2 - b1) * 1.0 / (k1 - k2)
    y = k1 * x * 1.0 + b1 * 1.0
    return [x, y]

def point_line(point,line):
    x1 = line[0]  
    y1 = line[1]
    x2 = line[2]
    y2 = line[3]

    x3 = point[0]
    y3 = point[1]

    k1 = (y2 - y1)*1.0 /(x2 -x1) 
    b1 = y1 *1.0 - x1 *k1 *1.0
    k2 = -1.0/k1
    b2 = y3 *1.0 -x3 * k2 *1.0
    x = (b2 - b1) * 1.0 /(k1 - k2)
    y = k1 * x *1.0 +b1 *1.0
    return [x,y]

def point_point(point_1,point_2):
    x1 = point_1[0]
    y1 = point_1[1]
    x2 = point_2[0]
    y2 = point_2[1]
    distance = ((x1-x2)**2 +(y1-y2)**2)**0.5
    return distance

def estimate(frame, transform, plfd_backbone):
    height, width = frame.shape[:2]
    model_test = AntiSpoofPredict(0)
    image_bbox = model_test.get_bbox(frame)
    
    x1 = image_bbox[0]
    y1 = image_bbox[1]
    x2 = image_bbox[0] + image_bbox[2]
    y2 = image_bbox[1] + image_bbox[3]
    w = x2 - x1
    h = y2 - y1

    size = int(max([w, h]))
    cx = x1 + w/2
    cy = y1 + h/2
    x1 = cx - size/2
    x2 = x1 + size
    y1 = cy - size/2
    y2 = y1 + size

    dx = max(0, -x1)
    dy = max(0, -y1)
    x1 = max(0, x1)
    y1 = max(0, y1)

    edx = max(0, x2 - width)
    edy = max(0, y2 - height)
    x2 = min(width, x2)
    y2 = min(height, y2)

    cropped = frame[int(y1):int(y2), int(x1):int(x2)]
    if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
        cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)
    # print(cropped.shape)
    cropped = cv2.resize(cropped, (112, 112))

    input = cv2.resize(cropped, (112, 112))
    input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    input = transform(input).unsqueeze(0).to(device)
    _, landmarks = plfd_backbone(input)
    pre_landmark = landmarks[0]
    pre_landmark = pre_landmark.cpu().detach().numpy().reshape(-1, 2) * [112, 112]
    point_dict = {}
    i = 0
    for (x,y) in pre_landmark.astype(np.float32):
        point_dict[f'{i}'] = [x,y]
        i += 1

    #yaw
    point1 = [get_num(point_dict, 1, 0), get_num(point_dict, 1, 1)]
    point31 = [get_num(point_dict, 31, 0), get_num(point_dict, 31, 1)]
    point51 = [get_num(point_dict, 51, 0), get_num(point_dict, 51, 1)]
    crossover51 = point_line(point51, [point1[0], point1[1], point31[0], point31[1]])
    yaw_mean = point_point(point1, point31) / 2
    yaw_right = point_point(point1, crossover51)
    yaw = (yaw_mean - yaw_right) / yaw_mean
    yaw = int(yaw * 71.58 + 0.7037)

    #pitch
    pitch_dis = point_point(point51, crossover51)
    if point51[1] < crossover51[1]:
        pitch_dis = -pitch_dis
    pitch = int(1.497 * pitch_dis + 18.97)

    #roll
    roll_tan = abs(get_num(point_dict,60,1) - get_num(point_dict,72,1)) / abs(get_num(point_dict,60,0) - get_num(point_dict,72,0))
    roll = math.atan(roll_tan)
    roll = math.degrees(roll)
    if get_num(point_dict, 60, 1) > get_num(point_dict, 72, 1):
        roll = -roll
    roll = int(roll)
    
    return yaw, pitch, roll

def optical_flow(one, two):
    one_g = cv2.cvtColor(one, cv2.COLOR_RGB2GRAY)
    two_g = cv2.cvtColor(two, cv2.COLOR_RGB2GRAY)
    # obtain dense optical flow paramters
    flow = cv2.calcOpticalFlowFarneback(one_g, two_g, flow=None,
                                        pyr_scale=0.5, levels=1, winsize=15,
                                        iterations=2,
                                        poly_n=5, poly_sigma=1.1, flags=0)
    # convert from cartesian to polar
    mag = flow[..., 0]
    return abs(np.mean(mag))

def main():

    file_path = "../vid_3DDFA_raw_all/"+args["Videopath"]+".MOV"
    print("==================Initializing Face Tracking================")
    if not os.path.isdir('../frames_3DDFA/'+args["Videopath"]):
        os.mkdir('../frames_3DDFA/'+args["Videopath"])
    
    if len(os.listdir('../frames_3DDFA/'+args["Videopath"])) > 400:
        return
    print(file_path)
    
    fvs = FileVideoStream(file_path).start()

    detector = dlib.get_frontal_face_detector()
    stabilizer = VidStab()

    tracker = ECOTracker(True)
    
    is_frame = True
    faces = ()
    detected = False

    checkpoint = torch.load("./checkpoint/snapshot/checkpoint.pth.tar", map_location=device)
    plfd_backbone = PFLDInference().to(device)
    plfd_backbone.load_state_dict(checkpoint['plfd_backbone'])
    plfd_backbone.eval()
    plfd_backbone = plfd_backbone.to(device)
    transform = transforms.Compose([transforms.ToTensor()])
    videoCapture = cv2.VideoCapture(file_path)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("fps:",fps,"size:",size)

    # success,frame = videoCapture.read()
    rolls = []
    for _ in range(5):
        success,frame = videoCapture.read()
        frame = imutils.resize(frame, width=800)
        try:
            _, _, roll = estimate(frame, transform, plfd_backbone)
        except:
            roll = 0
        rolls.append(roll)
    ROTATION_ANGLE = -int(np.average(rolls))
    # ROTATION_ANGLE = 15
    boarder = 0
    
    while not detected and success:
        # frame = fvs.read()
        frame = imutils.rotate_bound(frame,ROTATION_ANGLE)
        # cv2.imshow("face",frame)
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord("q"):
        #     break
        boarder = int((frame.shape[1] - 1080)/2)
        frame = imutils.resize(frame, width=800)
        faces = detector(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not len(faces):
            # print("No")
            detected = False
            success,frame = videoCapture.read()
            continue
        # face = [faces[0].left(),faces[0].top(),faces[0].right()-faces[0].left(),faces[0].bottom()-faces[0].top()]
        face = [faces[0].left(), faces[0].top(), faces[0].width(), faces[0].height()]
        # print(face[2])
        if face[0]<0 or face[1]<0 or face[2] > 650:
            detected = False
            success,frame = videoCapture.read()
            continue
        detected = True
        print("hot start; ", face)
        
        initbox = face

        tracker.init(frame,face)
    # estimate natural roll
    
        
        
    fn = 0
    prev_frame = None
    frame_count = 0
    rys = []
    drys = []
    pyaw, ppitch, proll = 0, 0, 0
    prx,pry,prz = 0, 0, 0
    
    # success,frame0 = videoCapture.read()
    while success:
        # print(boarder)
        frame = imutils.rotate_bound(frame,ROTATION_ANGLE)

        fn += 1
        frame = imutils.resize(frame, width=800)

        bbox = tracker.update(frame,True,True)
        
        if not boarder or boarder < 0:
            try:
                yaw, pitch, roll = estimate(frame, transform, plfd_backbone)
                pyaw, ppitch, proll = yaw, pitch, roll
            except:
                yaw, pitch, roll = pyaw, ppitch, proll
        else:            
            try:
                yaw, pitch, roll = estimate(frame[boarder:-boarder,boarder:-boarder], transform, plfd_backbone)
                pyaw, ppitch, proll = yaw, pitch, roll
            except:
                yaw, pitch, roll = pyaw, ppitch, proll
        
        rx, ry, rz = pitch, yaw, roll
        
        # print(ry,rx,rz)
        drys.append(abs(ry-pry))
        if len(drys) == 6:
            # drys.sort()
            drys.pop(0)
        
        rys.append(abs(ry))
        if len(rys) == 6:
            rys.sort()
            rys.pop(-1)
        prx,pry,prz = rx,ry,rz

        
        
        
        frame = imutils.resize(frame, width=400)
        x, y, w, h = bbox
        
        w,h = initbox[2],initbox[3]
        x1=max(0,int(y/2-20))
        x2=min(int((y+h)/2+20),frame.shape[0])
        y1=max(0,int(x/2-20))
        y2=min(int((x+w)/2+20),frame.shape[1])
        out_frame = frame[x1:x2,y1:y2]


        canvas = 225*np.ones((int(h/2)+41,int(h/2)+41,3), np.uint8)
        canvas[max(0,-int(y/2-20)):(max(0,-int(y/2-20))+out_frame.shape[0]),max(0,-int(x/2-20)):(max(0,-int(x/2-20))+out_frame.shape[1])] = out_frame
        pts1 = np.float32([[0,0],[0,canvas.shape[0]],[canvas.shape[0],canvas.shape[0]],[canvas.shape[0],0]])
        pts2 = np.float32([[0,0],[0,250],[250,250],[250,0]])

        M = cv2.getPerspectiveTransform(pts1,pts2)

        dst = cv2.warpPerspective(canvas,M,(250,250))

        # Display the resulting frame
        dst = stabilizer.stabilize_frame(input_frame=dst,smoothing_window=5,border_type='replicate')
        
        # print(abs(ry) > 15, abs(rz) > 15, sum(rys) > 60, sum(drys) > 10)
        if abs(ry) > 18 or abs(rz) > 15 or sum(rys) > 80 or sum(drys) > 15:
            # print("large motion")
            success,frame = videoCapture.read()
            continue

        
        
        
        frame_count += 1
        
        if frame_count < 6:
            prev_frame = frame
            success,frame = videoCapture.read()
            continue
        
        if optical_flow(frame,prev_frame) < 0.02:
            prev_frame = frame
            success,frame = videoCapture.read()
            continue
        
        dst = dst[13:237,13:237]
        # cv2.imshow("face",dst)
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord("q"):
        #     break
        cv2.imwrite('../frames_3DDFA/'+args["Videopath"]+'/'+str(fn)+'.jpg',dst)
        prev_frame = frame
        
        success,frame = videoCapture.read()
            
    cv2.destroyAllWindows()
    fvs.stop()
    return

if __name__ == '__main__':

    a = argparse.ArgumentParser()
    a.add_argument("--Videopath","-v")
    args = vars(a.parse_args())
    main()