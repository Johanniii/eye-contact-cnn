## Package import
import torch
import torchvision.io
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim

import time
import copy
import pandas as pd
from eye_contact_frames import eye_contact_frames


## to load a model:
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth"))

## label preparation
files = pd.read_excel('annotations/video_annotations.xlsx')
annotation_files = [(file, end_frame, train_val) for file, end_frame, train_val in zip(files.Annotation_file, files.annot_frame_end, files.train_val)]

train_labels = []
val_labels = []

# get labels of one file
def get_labels(annotation_file):
    label = eye_contact_frames(f'annotations/{annotation_file[0]}.xml', int(annotation_file[1]))

    if annotation_file[2] == 'train':

        train_labels.append(label)
        return train_labels
    
    elif annotation_file[2] == 'val':

        val_labels.append(label)
        return val_labels
    
# read all labels
for file in annotation_files:
    get_labels(file)

## helper function for reading frames
def read_frames(video_path, frames):
    end_pts = frames/30 # 30 fps
    video_frames, _, _ = torchvision.io.read_video(video_path, end_pts=end_pts, pts_unit='sec')

    # Return the video frames
    return video_frames


# Transformation and augmentation:
# with Compose:
transforms = {
    'train': transforms.Compose([
        # augmentation:
        transforms.RandomResizedCrop(224), # crop images
        transforms.RandomHorizontalFlip(), # flip images
        # normalization:
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # mean, deviation for R,G,B; calculated on IMGnet
    ]),
    'val': transforms.Compose([
        transforms.Resize(256), # no augmentation, just resize and crop on validation data
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

## Class for Video Dataset (frames are missing for read_frames)
class CustomVideoDataset(Dataset):

    def __init__(self, video_dir, labels, transform=None):
        self.video_dir = video_dir
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        video_dir = self.video_dir[index]
        label = self.labels[index]

        # Load and transform video frames
        frames = read_frames(video_dir)
        if self.transform:
            frames = self.transform(frames)

        return frames, label
    
    def __len__(self):
        video_dir = self.video_dir
        
        return len(read_frames(video_dir))
    

## Create training and validation datasets
train_dataset = CustomVideoDataset('videos/train_val/train', train_labels, transform=transforms)
val_dataset = CustomVideoDataset('videos/train_val/val', val_labels, transform=transforms)

## Prepare data with data loaders
batch_size = 50
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


#### the following parts are so far unedited

## Define loss function
criterion = nn.BCELoss()

## Define optimizer
learning_rate = 0.001
weight_decay = 0.0001
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

## set device
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
    )

## helper function for model training
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
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
            for inputs, labels in dataloaders[phase]:
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

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

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
    return model

## Validation loop
model.eval()
val_loss = 0.0

with torch.no_grad():
    for frames, labels in val_loader:
        frames = frames.to(device)
        labels = labels.to(device)

        outputs = model(frames)
        loss = criterion(outputs, labels)

        val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f"Validation Loss: {val_loss:.4f}")

## Save model
torch.save(model.state_dict(), "model_new.pth")