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


class Trainer:
    
    def __init__(self,
                 image_datasets,
                 num_epochs=25,
                 batch_size=8,
                 visualize=False
                 ):
        
        self.num_epochs = num_epochs
        self.model = None
        self.dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
                                                           batch_size=batch_size,
                                                           shuffle=True, 
                                                           num_workers=0)
                            for x in ['train', 'val']}
        self.dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        

        if visualize:
            class_names = ['neg', 'pos']
            # Get a batch of training data and visualize it
            inputs, classes = next(iter(self.dataloaders['train']))
            # Make a grid from batch
            out = torchvision.utils.make_grid(inputs)
            self.imshow(out, title=[class_names[x] for x in classes])
            
    @staticmethod
    def imshow(inp, title=None):
        """Imshow for Tensor."""
        print('showing image')
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated
            
    def train_model(self, model, criterion, optimizer, scheduler):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('on device:', device)
        
        model = model.to(device)
        
        since = time.time()
    
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        
        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)
    
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode
    
                running_loss = 0.0
                running_corrects = 0
    
                # Iterate over data.
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
    
                    # zero the parameter gradients
                    optimizer.zero_grad()
    
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
    
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
    
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()
    
                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]
    
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    
                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
    
            print()
    
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')
    
        # load best model weights
        model.load_state_dict(best_model_wts)
        self.model = model


class Resize_with_pad:
    
    def __init__(self, w=300, h=300):
        self.w = w
        self.h = h

    def __call__(self, image):

        w_1, h_1 = image.size
        ratio_f = self.w / self.h
        ratio_1 = w_1 / h_1


        # check if the original and final aspect ratios are the same within a margin
        if round(ratio_1, 2) != round(ratio_f, 2):

            # padding to preserve aspect ratio
            hp = int(w_1/ratio_f - h_1)
            wp = int(ratio_f * h_1 - w_1)
            if hp > 0 and wp < 0:
                hp = hp // 2
                image = F.pad(image, (0, hp, 0, hp), 0, "constant")
                return F.resize(image, [self.h, self.w])

            elif hp < 0 and wp > 0:
                wp = wp // 2
                image = F.pad(image, (wp, 0, wp, 0), 0, "constant")
                return F.resize(image, [self.h, self.w])

        else:
            return F.resize(image, [self.h, self.w])

        
class ImageDataset(Dataset):
    """ construct dataset for image processing """
    def __init__(self, file_list, labels, transform):
        self.file_list = file_list
        self.labels = labels
        self.transform = transform
        
    def __getitem__(self, idx):
        """ read img from file, then transform """
        file = self.file_list[idx]
        img = Image.open(file)
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label
    
    def __len__(self):
        return len(self.labels)


class Preprocessor:
    
    def __init__(self,
                  data_path,
                 img_height=300,
                 img_width=300
                 ):
        
        self.img_height = img_height
        self.img_width = img_width
        self.data_path = data_path
        
        self.init_transform = {'train': 
                      transforms.Compose(
        [
         Resize_with_pad(img_height,img_width),
         transforms.RandomResizedCrop(300,scale=(0.9, 1.0),ratio=(1,1)),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
         ]
        ),
        'val':
            transforms.Compose(
        [
         Resize_with_pad(img_height,img_width),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
         ]
        )}

    
        self.imgdir_path = {x: pathlib.Path(os.path.join(self.data_path, x))
                for x in ['train', 'val']}
        # self.imgdir_path = {x: pathlib.Path(x)
        #        for x in ['train', 'val']}

        self.file_list = {x: sorted([str(path) for path in self.imgdir_path[x].glob('*.jpg')])
             for x in ['train', 'val']}
        
        print(self.imgdir_path)
        print(self.file_list)

    def get_data_labels(self, pos_examples):

        self.labels = {x:[1 if os.path.basename(fn) in pos_examples else 0 for fn in self.file_list[x]]
          for x in ['train', 'val']}
        
        self.image_datasets = {x: ImageDataset(self.file_list[x], self.labels[x], self.init_transform[x])
                         for x in ['train', 'val']}
        
        print(self.image_datasets['train'].file_list)


if __name__ == '__main__':
    #################################### HPs
    
    num_epochs = 5
    lr = 0.001
    momentum = 0.9
    step_size = 7
    gamma = 0.1
    batch_size = 8
    data_path = 'data'
    
    
    #################################### preprocessing
    
    pos_examples = {'IMG_2930.JPG', 'IMG_3176.JPG', 'IMG_2594.JPG', 'IMG_2492.JPG', 'IMG_8189.JPG',
     'IMG_2902.JPG', 'IMG_3170.JPG', 'IMG_2079.JPG', 'IMG_2951.JPG', 'IMG_3200.JPG',
     'IMG_3171.JPG', 'IMG_2327.JPG', 'IMG_3056.JPG', 'IMG_9052.JPG', 'IMG_0913.JPG', 
     'IMG_8721.JPG', 'IMG_8730.JPG', 'IMG_3135.JPG'}
    
    pos_examples = ['h_' + i.split('.')[0] + '.jpg' for i in pos_examples]
    print(pos_examples)
    
    preproc = Preprocessor(data_path,
                           img_height=300,
                           img_width=300)
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