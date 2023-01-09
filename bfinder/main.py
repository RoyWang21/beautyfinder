import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
import pathlib
import copy
from pathlib import Path

from config import config
from train import Preprocessor, Trainer
import argparse
import json
import utils

if __name__ == '__main__':
    args_fp = "config/args.json"
    args = argparse.Namespace(**utils.load_dict(filepath=args_fp))
    data_path = config.DATA_DIR
    
    print(args)
    
    img_height = 300
    img_width = 300
    
    num_epochs = 5
    lr = 0.001
    momentum = 0.9
    step_size = 7
    gamma = 0.1
    batch_size = 8
    
    #################################### preprocessing
    
    pos_examples = {'IMG_2930.JPG', 'IMG_3176.JPG', 'IMG_2594.JPG', 'IMG_2492.JPG', 'IMG_8189.JPG',
     'IMG_2902.JPG', 'IMG_3170.JPG', 'IMG_2079.JPG', 'IMG_2951.JPG', 'IMG_3200.JPG',
     'IMG_3171.JPG', 'IMG_2327.JPG', 'IMG_3056.JPG', 'IMG_9052.JPG', 'IMG_0913.JPG', 
     'IMG_8721.JPG', 'IMG_8730.JPG', 'IMG_3135.JPG'}
    
    pos_examples = ['h_' + Path(i).stem + '.jpg' for i in pos_examples]
    print(pos_examples)
    
    preproc = Preprocessor(data_path,
                           img_height=img_height,
                           img_width=img_width)
    preproc.get_data_labels(pos_examples)
    
    #################################### model
    
    model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False
    
    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)
    
    #################################### training
    
    trainer = Trainer(preproc.image_datasets,
                 num_epochs=num_epochs,
                 batch_size=batch_size,
                 visualize=True)
    
    criterion = nn.CrossEntropyLoss()
    
    # Only parameters of final layer are being optimized
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=lr, momentum=momentum)
    
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=step_size, gamma=gamma)
    
    
    trainer.train_model(model_conv, criterion, optimizer_conv,
                              exp_lr_scheduler)
    
    
# imgdir_path = pathlib.Path('.')
# #('./data')
# file_list = sorted([str(path) for path in imgdir_path.glob('*.JPG')])
# file_list = [fn.split('.')[0] for fn in file_list]
# print(file_list)
# cropper = Cropper()

# file_list = ['IMG_2006',
#   'IMG_2416']


# modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
# configFile = "deploy.prototxt.txt"
# net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# split_schedule = [[1,1],[1,2],[2,1],[2,3],[3,2], None]

# # file_list = ['IMG_8714']
# from split_image import split_image, reverse_split
# for file in file_list:
#     face_found = False
#     idx = 0
#     while not face_found:
#         split = split_schedule[idx]
#         if not split: break
#         print('split:', split)
#         split_image(file+'.JPG', split[0], split[1], should_square=False, should_cleanup=False)
#         sub_file_list = [file+'_'+str(i)+'.JPG' for i in range(split[0]*split[1])]
#         for sub_file in sub_file_list:
#             print(f'detecting face on {sub_file}')
    
#             img = cv2.imread(sub_file)
#             h, w = img.shape[:2]
#             print(h, w)
#             blob = cv2.dnn.blobFromImage(
#                 cv2.resize(img, (300, 300)), 
#                 # img,
#                 1.0,
#                 # (w, h),
#                 (300,300),
#                 (104.0, 117.0, 123.0))
#             net.setInput(blob)
#             faces = net.forward()
#             #to draw faces on image
#             for i in range(faces.shape[2]):
#                     confidence = faces[0, 0, i, 2]
#                     if confidence > 0.5:
#                         face_found = True
#                         print('--face detected!')
#                         box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
#                         (x, y, x1, y1) = box.astype("int")
#                         img_box = cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)
#                         break
#         if not face_found: print('--face not found, splitting image and retrying...')
#         idx += 1
#         reverse_split(sub_file_list, split[0], split[1], file+'_rev.png', should_cleanup=True)
        
#     if face_found:
#         cv2.startWindowThread()
#         cv2.namedWindow("Image_with_face")
#         cv2.imshow('Image_with_face', img_box) 
#         cv2.waitKey(800)
#         cv2.destroyAllWindows()
        
