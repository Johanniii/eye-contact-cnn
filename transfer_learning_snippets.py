import torch
import torchvision.io
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim


### to load a model:
# model = NeuralNetwork().to(device)
# model.load_state_dict(torch.load("model.pth"))

### Try out reading data
torchvision.io.read_video('videos/train_val/val/5.mp4')


### Helper function for reading data
def read_frames(video_path):
    video_frames, _, _ = torchvision.io.read_video(video_path, end_pts=5.0)
    #video_frames = video_frames[0:end_frame]

    # Return the video frames
    return video_frames



### Transformation and augmentation:
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
# Sequential alternative:
transforms = {
    'train': torch.nn.Sequential(
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(), # flip images
        # normalization:
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ),
    'val': torch.nn.Sequential(
        transforms.Resize(256), # no augmentation, just resize and crop on validation data
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    )
}


### Read whole Video Data set with labels
class CustomVideoDataset(Dataset):

    def __init__(self, video_dir, labels, transform=None):
        self.video_dir = video_dir
        self.labels = labels # needs to be edited - will they be stored or will eye_contact_frames.py be called?
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


### Create training and validation datasets
train_dataset = CustomVideoDataset(video_dir, train_labels, transform=transforms)
val_dataset = CustomVideoDataset(video_dir, val_labels, transform=transforms)

### def GetDataLoader(path, batch_size, num_workers):
#     dataset = VideoDataset(path)
#     dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

### Prepare data with data loaders
batch_size = 70
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
optimizers = ['Adam', 'SGD']
learning_rates = [0.001, 0.01, 0.1]
weight_decays = [0.0001, 0.001, 0.01]

### Helper functions for optimization
def get_optimizer(optimizer, learning_rate, weight_decay):
    pass

def evaluate(model):
    pass

def record_performance(optimizer, learning_rate, weight_decay, accuracy):
    pass

### optimization grid
for optimizer in optimizers:
    for learning_rate in learning_rates:
        for weight_decay in weight_decays:
            # Create the optimizer instance with the current hyperparameters
            optimizer_instance = get_optimizer(optimizer, learning_rate, weight_decay) # helper function

            # Train and evaluate the model using the current optimizer and hyperparameters
            train(model, optimizer_instance) # helper function
            accuracy = evaluate(model) # helper function

            # Record and compare the performance for analysis
            record_performance(optimizer, learning_rate, weight_decay, accuracy) # helper function


# # Define loss function
# criterion = nn.BCELoss()

# # Define optimizer
# learning_rate = 0.001
# weight_decay = 0.0001
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# set device
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
    )


### Training loop
model.to(device)
criterion.to(device)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (frames, labels) in enumerate(train_loader):
        frames, labels = frames.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(frames)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")


### Alternatives?
# def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
#     since = time.time()

#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0

#     for epoch in range(num_epochs):
#         print(f'Epoch {epoch}/{num_epochs - 1}')
#         print('-' * 10)

#         # Each epoch has a training and validation phase
#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 model.train()  # Set model to training mode
#             else:
#                 model.eval()   # Set model to evaluate mode

#             running_loss = 0.0
#             running_corrects = 0

#             # Iterate over data.
#             for inputs, labels in dataloaders[phase]:
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)

#                 # zero the parameter gradients
#                 optimizer.zero_grad()

#                 # forward
#                 # track history if only in train
#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs)
#                     _, preds = torch.max(outputs, 1)
#                     loss = criterion(outputs, labels)

#                     # backward + optimize only if in training phase
#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()

#                 # statistics
#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)
#             if phase == 'train':
#                 scheduler.step()

#             epoch_loss = running_loss / dataset_sizes[phase]
#             epoch_acc = running_corrects.double() / dataset_sizes[phase]

#             print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

#             # deep copy the model
#             if phase == 'val' and epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 best_model_wts = copy.deepcopy(model.state_dict())

#         print()

#     time_elapsed = time.time() - since
#     print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
#     print(f'Best val Acc: {best_acc:4f}')

#     # load best model weights
#     model.load_state_dict(best_model_wts)
#     return model
# odel_ft = models.resnet18(weights='IMAGENET1K_V1')
# num_ftrs = model_ft.fc.in_features
# # Here the size of each output sample is set to 2.
# # Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
# model_ft.fc = nn.Linear(num_ftrs, 2)

# model_ft = model_ft.to(device)

# criterion = nn.CrossEntropyLoss()

# # Observe that all parameters are being optimized
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# # Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
# model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
#                        num_epochs=25)
# model_conv = torchvision.models.resnet18(weights='IMAGENET1K_V1')
# for param in model_conv.parameters():
#     param.requires_grad = False

# # Parameters of newly constructed modules have requires_grad=True by default
# num_ftrs = model_conv.fc.in_features
# model_conv.fc = nn.Linear(num_ftrs, 2)

# model_conv = model_conv.to(device)

# criterion = nn.CrossEntropyLoss()

# # Observe that only parameters of final layer are being optimized as
# # opposed to before.
# optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# # Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


# model_conv = train_model(model_conv, criterion, optimizer_conv,
#                          exp_lr_scheduler, num_epochs=25)






### Validation loop
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


### Save model
torch.save(model.state_dict(), "model_new.pth")