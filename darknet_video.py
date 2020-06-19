from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.utils.data as data
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io, draw

###################################################
class Image_Dataset(data.Dataset):
    
    def __init__(self, image_list, transform=None):    
        self.image_list = image_list
        self.transform = transform
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        image = self.image_list[idx]
        pil_image = Image.fromarray(image, mode = "RGB")
        img_transformed = self.transform(pil_image)

        return img_transformed
####################################################
def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)

mask_model = models.resnet50(pretrained=True)
mask_model.fc = torch.nn.Sequential(torch.nn.Linear(2048, 1024),
                                 torch.nn.BatchNorm1d(1024),
                                 torch.nn.ReLU(),
                                 torch.nn.Dropout(0.2),
                                 torch.nn.Linear(1024, 512),
                                 torch.nn.BatchNorm1d(512),
                                 torch.nn.Dropout(0.6),
                                 torch.nn.Linear(512, 2),
                                 torch.nn.LogSoftmax(dim=1))

train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

mask_model.to(device)

def load_mask_wt(path = '/content/drive/My Drive/equalaf4.pth'):
    mask_model.load_state_dict(torch.load(path))
    
font_scale = 0.35
thickness = 1
blue = (0,0,255)
green = (0,255,0)
red = (255,0,0)
font=cv2.FONT_HERSHEY_COMPLEX
################################################ 

def cvDrawBoxes(detections, img, mask_wt_path = "/content/drive/My Drive/equalaf4.pth"):
    load_mask_wt(mask_wt_path)
    mask_model.eval()
    BATCH_SIZE = 0
    result = []
    ################################################################
    for detection in detections:
        x, y, w, h = detection[2][0], detection[2][1], detection[2][2], detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        
        detect_mask_img = img
        xCoord = int(x - w/2)
        yCoord = int(y - h/2)
        xi, yi, wi, hi = int(xCoord), int(yCoord), int(w), int(h)
        print(xi, yi, wi, hi)
        if (xi<0 or yi<0):
            xi = 0
            yi = 0
            if xi<0:
                wi = wi + xi
            if yi<0:
                hi = hi + yi
        
        detect_mask_img = detect_mask_img[yi:yi+hi, xi:xi+wi]
        result.append(detect_mask_img)
        BATCH_SIZE += 1
    #--------------------------------------------------
    comp = Image_Dataset(result, transform=train_transforms)
    test_loader = torch.utils.data.DataLoader(comp,
                            batch_size=BATCH_SIZE,
                                shuffle=False)

    print("accessing mask model")     
    prediction_list = []
    with torch.no_grad():
        print("load tl")
        for X in test_loader:
                    #X = X.cuda()
            print(device)
            ans = mask_model(X.cuda())
            '''if device=="cuda":
                ans = mask_model(X.cuda())
            else:
                ans = mask_model(X)'''
            print("make prediction")
            _, maximum = torch.max(ans.data, 1)
            print(maximum.tolist())
            prediction_list = maximum.tolist()
    print("predictions: ", prediction_list)       
        
    #-----------------------------------------------------    
    i=0
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        
        detect_mask_img = img
        xCoord = int(x - w/2)
        yCoord = int(y - h/2)
        xi, yi, wi, hi = int(xCoord), int(yCoord), int(w), int(h)
        print(xi, yi, wi, hi)

        if (xi<0 or yi<0):
            xi = 0
            yi = 0
            if xi<0:
                wi = wi + xi
            if yi<0:
                hi = hi + yi
        
        prediction = prediction_list[i]

        if prediction == 0:
            cv2.putText(img, "No Mask", (xi,yi - 10), font, font_scale, red, thickness)
            boxColor = red
        elif prediction == 1:
            cv2.putText(img, "Mask", (xi,yi - 10), font, font_scale, green, thickness)
            boxColor = green
        print("prediction : " + str(prediction))
        cv2.rectangle(img, pt1, pt2, boxColor, 1)
        i+=1
        
        ################################################################
    return img

netMain = None
metaMain = None
altNames = None


def YOLO(video_path = '/content/mask_footage.mp4', configPath = "cfg/custom-yolov4-detector.cfg", weightPath = "/content/custom-yolov4-detector_best.weights", metaPath = "data/obj.data", mask_wt_path = "/content/drive/My Drive/equalaf4.pth"):

    global metaMain, netMain, altNames
    '''configPath = "./cfg/yolov4.cfg"
    weightPath = "./yolov4.weights"
    metaPath = "./cfg/coco.data"'''
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(video_path)
    cap.set(3, 1280)
    cap.set(4, 720)
    out = cv2.VideoWriter(
        "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 27.0,
        (darknet.network_width(netMain), darknet.network_height(netMain)))
    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
    frame_no = 0
    total_time = 0
    while True:
        try:
            frame_no += 1
            prev_time = time.time()
            ret, frame_read = cap.read()
            frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb,
                                       (darknet.network_width(netMain),
                                        darknet.network_height(netMain)),
                                       interpolation=cv2.INTER_LINEAR)

            darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())
           
        
            detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
            image = cvDrawBoxes(detections, frame_resized, mask_wt_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            out.write(image)
            print(1/(time.time()-prev_time))
            total_time += time.time()-prev_time
            #io.imshow(image)
            #io.show()
            cv2.waitKey(3)
        except:
            break
    if(total_time!=0):
        fps = frame_no / total_time
        print("FPS = ", fps)
      
    cap.release()
    out.release()

if __name__ == "__main__":
    YOLO(video_path = '/content/mask_footage.mp4', configPath = "cfg/custom-yolov4-detector.cfg", weightPath = "/content/custom-yolov4-detector_best.weights", metaPath = "data/obj.data", mask_wt_path = "/content/drive/My Drive/equalaf4.pth")
