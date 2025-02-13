import os
from PIL import Image
import cv2
import numpy as np
import torch 
import torchvision.transforms as transforms
from tqdm import tqdm

transform=transforms.Compose([
                                      transforms.Resize(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])
                                      ])
frame_pth = "./frames/"
for f in tqdm(os.listdir(frame_pth)):
    img_lists = os.listdir(frame_pth+f)
    if len(img_lists) < 64*7:
        print("skip "+f)
        continue
    img_lists.sort(key=lambda x: int(x[:-4]))
    
    if not os.path.exists('./7seg/video_segment_fix/'+f):
        os.mkdir('./7seg/video_segment_fix/'+f)
    for i in range(7):
        frames = []
        for j in range(64):
            img = Image.open(os.path.join(frame_pth+f, img_lists[i*64+j])).convert("RGB")
            feat = transform(img)
            frames.append(feat)
        frames = torch.stack(frames, dim=0)
        np.save('./7seg/video_segment_fix/'+f+'/'+str(i+1).zfill(2)+'.npy', frames)
        
