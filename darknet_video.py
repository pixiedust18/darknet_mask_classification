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


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

################################################
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
    final_img = img
    
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        
        ################################################################
        
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
        #io.imshow(detect_mask_img)
        #io.show()
        pil_image = Image.fromarray(detect_mask_img, mode = "RGB")
        pil_image = train_transforms(pil_image)
        img_modif = pil_image.unsqueeze(0)
                            
        print("accessing mask model")            
        result = mask_model(img_modif)
        _, maximum = torch.max(result.data, 1)
        prediction = maximum.item()
       
        '''if prediction == 0:
                  if mask_present_label == True:
                    cv2.putText(img, "No Mask", (x,y - 10), font, font_scale, red, thickness)
                    print("Label print", mask_present_label)
                  else:
                    print("Label print", mask_present_label)
                  print("No mask")
            boxColor = red
        elif prediction == 1:
                  if mask_present_label == True:
                    cv2.putText(img, "Mask", (x,y - 10), font, font_scale, green, thickness)
                    print("Label print", mask_present_label)
                  else:
                    print("Label print", mask_present_label)
                  print("Mask")
                  boxColor = green'''
        
        if prediction == 0:
            cv2.putText(final_img, "No Mask", (xi,yi - 10), font, font_scale, red, thickness)
            boxColor = red
        elif prediction == 1:
            cv2.putText(final_img, "Mask", (xi,yi - 10), font, font_scale, green, thickness)
            boxColor = green
        print("prediction : " + str(prediction))
        cv2.rectangle(final_img, pt1, pt2, boxColor, 1)
        '''cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)'''
        
        ################################################################
    return final_img


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
    while True:
        try:
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
            #io.imshow(image)
            #io.show()
            cv2.waitKey(3)
        except:
            break;
      
    cap.release()
    out.release()

if __name__ == "__main__":
    YOLO(video_path = '/content/mask_footage.mp4', configPath = "cfg/custom-yolov4-detector.cfg", weightPath = "/content/custom-yolov4-detector_best.weights", metaPath = "data/obj.data", mask_wt_path = "/content/drive/My Drive/equalaf4.pth")
