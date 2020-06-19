""" 
Credit:       https://github.com/JadHADDAD92/covid-mask-detector
Call:         python detectMask-mp4.py --output crowd-1.mp4 ./checkpoints/model.ckpt C.mp4
Description:  Detect people wearing masks in videos
Remarks:      1.  Confidence Threshold is key to FaceDetection. Algorithm is still not picking up more faces to process in frame.
              2.  Putting CHINESE wordings on LABELS. This means the code can display CHINESE characters
              3.  Try to use Facenet to replace the FaceDetection - https://github.com/timesler/facenet-pytorch
"""
from pathlib import Path
import click

import cv2
from cv2 import resize
from cv2.dnn import blobFromImage, readNetFromCaffe

import torch
import torch.nn.init as init
from torch import long, tensor, Tensor
from torch.nn import (Conv2d, CrossEntropyLoss, Linear, MaxPool2d, ReLU, Sequential)
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor

import numpy as np

from skvideo.io import FFmpegWriter, vreader

from typing import Dict, List, Union

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from PIL import Image, ImageDraw, ImageFont

class FaceDetectorException(Exception):
    """ generic default exception
    """
class FaceDetector:
    """ Face Detector class - Hard code confidence Threshold as 0.25 
    """
    def __init__(self, prototype: Path=None, model: Path=None,
                 confidenceThreshold: float=0.5):
        self.prototype = prototype
        self.model = model
        self.confidenceThreshold = confidenceThreshold
        if self.prototype is None:
            raise FaceDetectorException("must specify prototype '.prototxt.txt' file "
                                        "path")
        if self.model is None:
            raise FaceDetectorException("must specify model '.caffemodel' file path")
        self.classifier = readNetFromCaffe(str(prototype), str(model))
    
    def detect(self, image):
        """ detect faces in image
        """
        net = self.classifier
        height, width = image.shape[:2]
        blob = blobFromImage(resize(image, (300, 300)), 1.0,
                             (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < self.confidenceThreshold:
                continue
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            startX, startY, endX, endY = box.astype("int")
            faces.append(np.array([startX, startY, endX-startX, endY-startY]))
        return faces

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  #判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(
        "font/simsun.ttc", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


class MaskDataset(Dataset):
    """ Masked faces dataset
        0 = 'no mask'
        1 = 'mask'
    """
    def __init__(self, dataFrame):
        self.dataFrame = dataFrame
        
        self.transformations = Compose([
            ToPILImage(),
            Resize((100, 100)),
            ToTensor(), # [0, 1]
        ])
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            raise NotImplementedError('slicing is not supported')
        
        row = self.dataFrame.iloc[key]
        return {
            'image': self.transformations(cv2.imread(row['image'])),
            'mask': tensor([row['mask']], dtype=long), # pylint: disable=not-callable
        }
    
    def __len__(self):
        return len(self.dataFrame.index)

class MaskDetector(pl.LightningModule):
    """ MaskDetector PyTorch Lightning class
    """
    def __init__(self, maskDFPath: Path=None):
        super(MaskDetector, self).__init__()
        self.maskDFPath = maskDFPath
        
        self.maskDF = None
        self.trainDF = None
        self.validateDF = None
        self.crossEntropyLoss = None
        self.learningRate = 0.00001
        
        self.convLayer1 = convLayer1 = Sequential(
            Conv2d(3, 32, kernel_size=(3, 3), padding=(1, 1)),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2))
        )
        
        self.convLayer2 = convLayer2 = Sequential(
            Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2))
        )
        
        self.convLayer3 = convLayer3 = Sequential(
            Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1), stride=(3,3)),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2))
        )
        
        self.linearLayers = linearLayers = Sequential(
            Linear(in_features=2048, out_features=1024),
            ReLU(),
            Linear(in_features=1024, out_features=2),
        )
        
        # Initialize layers' weights
        for sequential in [convLayer1, convLayer2, convLayer3, linearLayers]:
            for layer in sequential.children():
                if isinstance(layer, (Linear, Conv2d)):
                    init.xavier_uniform_(layer.weight)
    
    def forward(self, x: Tensor): # pylint: disable=arguments-differ
        """ forward pass
        """
        out = self.convLayer1(x)
        out = self.convLayer2(out)
        out = self.convLayer3(out)
        out = out.view(-1, 2048)
        out = self.linearLayers(out)
        return out
    
   
    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), lr=self.learningRate)
    
@click.command(help="""
                    modelPath: path to model.ckpt\n
                    videoPath: path to video file to annotate
                    """)
@click.argument('modelpath')
@click.argument('videopath')
@click.option('--output', 'outputPath', type=Path,
              help='specify output path to save video with annotations')
@torch.no_grad()

def tagVideo(modelpath, videopath, outputPath=None):
    """ detect if persons in video are wearing masks or not
    """
    model = MaskDetector()
    model.load_state_dict(torch.load(modelpath)['state_dict'], strict=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    faceDetector = FaceDetector(
        prototype='./models/deploy.prototxt.txt',
        model='./models/res10_300x300_ssd_iter_140000.caffemodel',
    )

    transformations = Compose([
        ToPILImage(),
        Resize((100, 100)),
        ToTensor(),
    ])

    if outputPath:
        writer = FFmpegWriter(str(outputPath))

    # fontC = 'simsun.ttc'
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.FONT_HERSHEY_SIMPLEX
    cv2.namedWindow('main', cv2.WINDOW_NORMAL)
    labels = ['No mask', 'Mask']
    labelColor = [(255, 255, 255), (10, 255, 0)]  # Can have a different color for predicted with mask or without
    for frame in vreader(str(videopath)):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = faceDetector.detect(frame)
        for face in faces:
            xStart, yStart, width, height = face

            # clamp coordinates that are outside of the image
            xStart, yStart = max(xStart, 0), max(yStart, 0)

            # predict mask label on extracted face
            faceImg = frame[yStart:yStart+height, xStart:xStart+width]
            output = model(transformations(faceImg).unsqueeze(0).to(device))
            _, predicted = torch.max(output.data, 1)

            # draw face frame
            cv2.rectangle(frame,
                          (xStart, yStart),
                          (xStart + width, yStart + height),
                          (255, 255, 255),
                          thickness=2)

            # draw the prediction label in CHINESE

            imgNoMask = np.zeros([20,40,3],dtype=np.uint8)
            imgMask   = np.zeros([20,20,3],dtype=np.uint8)
            imgNoMask.fill(255)
            imgMask.fill(255)

            b,g,r,a = 0,0,0,0
            if predicted == 0:
                img = cv2ImgAddText(imgNoMask, "没有", 3, 3, (b,g,r), 15)
            else:
                img = cv2ImgAddText(imgMask, "有", 3, 3, (b,g,r), 15)
            img_height, img_width, _ = img.shape 
            frame[ yStart:yStart+img_height , xStart:xStart+img_width ] = img # Replace the top corner left with the image of Chinese words

            # Add the Prediction Label in ENGLISH according to the face frame

            # center text according to the face frame
            textSize = cv2.getTextSize(labels[predicted], font, 1, 2)[0]
            textX = xStart + width // 2 - textSize[0] // 2
            
            # draw prediction label
            cv2.putText(frame,
                        labels[predicted],
                        (textX+40, yStart+20),
                        font, 0.5, labelColor[predicted], 1)
            
        if outputPath:
            writer.writeFrame(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.imshow('main', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if outputPath:
        writer.close()
    cv2.destroyAllWindows()

# pylint: disable=no-value-for-parameter
if __name__ == '__main__':
    tagVideo()
