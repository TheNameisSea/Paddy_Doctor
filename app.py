#_____Imports_____#
# Timm
import sys
sys.path.append('../input/pytorch-image-models/pytorch-image-models-master')

# Settings
import warnings
import sklearn.exceptions
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

# General
from collections import defaultdict
import pandas as pd
import numpy as np
import os
import random
import gc
import cv2
import glob
gc.enable()
pd.set_option('display.max_columns', None)

# Augmentations
import albumentations
from albumentations.pytorch.transforms import ToTensorV2

# Deep Learning
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
import timm
import torch.nn as nn
import torch.nn.functional as F

# Seeds
RANDOM_SEED = 42

def seed_everything(seed=RANDOM_SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything()

# Device
device = torch.device('cpu')

#_____Data_____#
# Input
data_dir = './input'        # image folder
models_dir = './models'     # model folder
test_file_path = os.path.join(data_dir, 'sample_submission.csv')        # csv file with name of images
train_file_path = os.path.join(data_dir, 'train_5_folds.csv')       # csv file with labels and folds

image_ids = os.listdir('./upload')      # take name of all uploaded file
folder_len = len(image_ids)     # Take number of uploaded file

test_df = pd.read_csv(test_file_path)       # read template file
train_df = pd.read_csv(train_file_path)     # read label file

test_df = test_df[(test_df.index < folder_len)]     # remove images that could cause overflow
test_df['image_id'] = image_ids     # add name of uploaded file to template

test_df['image_path'] = test_df.apply(lambda row: './upload/' + row['image_id'], axis=1)        # create filepaths

# Labels
label2id = {'bacterial_leaf_blight': 0,
            'bacterial_leaf_streak': 1,
            'bacterial_panicle_blight': 2,
            'blast': 3,
            'brown_spot': 4,
            'dead_heart': 5,
            'downy_mildew': 6,
            'hispa': 7,
            'normal': 8,
            'tungro': 9}

id2label = {v: k for k, v in label2id.items()}

# Params
params = {
    'model': 'efficientnet_b3',
    'pretrained': False,
    'inp_channels': 3,
    'im_size': 300,
    'device': device,
    'batch_size': 85,
    'num_workers' : 0,
    'out_features': train_df['label'].nunique(),
    'dropout': 0.2,
    'num_fold': train_df['kfold'].nunique(),
    'debug': False,
}

# Transform
def get_test_transforms(DIM = params['im_size']):
    return albumentations.Compose(
        [
          albumentations.Resize(DIM,DIM),
          albumentations.Normalize(
              mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225],
          ),
          ToTensorV2(p=1.0)
        ]
    )

# Dataset and Dataloader
class PaddyDataset(Dataset):
    def __init__(self, images_filepaths, transform=None):
        self.images_filepaths = images_filepaths
        self.transform = transform

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.images_filepaths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)['image']
        
        return image

#_____Deep Learning_____#
# Neural Net
class PaddyNet(nn.Module):
    def __init__(self, model_name=params['model'], out_features=params['out_features'], inp_channels=params['inp_channels'],
                 pretrained=params['pretrained']):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, in_chans=inp_channels)
        out_channels = self.model.conv_stem.out_channels
        kernel_size = self.model.conv_stem.kernel_size
        stride = self.model.conv_stem.stride
        padding = self.model.conv_stem.padding
        bias = self.model.conv_stem.bias
        self.model.conv_stem = nn.Conv2d(inp_channels, out_channels,
                                          kernel_size=kernel_size, stride=stride,
                                          padding=padding, bias=bias)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        self.dropout = nn.Dropout(params['dropout'])
        self.fc = nn.Linear(n_features, out_features)
    
    def forward(self, image):
        embeddings = self.model(image)
        x = self.dropout(embeddings)
        output = self.fc(x)
        return output

# Prediction
pred_cols = []

for i, model_name in enumerate(glob.glob(models_dir + '/*.pth')):
    model = PaddyNet()
    model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))     # load model
    model = model.to(params['device'])
    model.eval()        # evaluate model for uses
    
    X_test = test_df['image_path']      # read filepaths

    # create a dataset of images and apply augmentation to the images
    test_dataset = PaddyDataset(
        images_filepaths=X_test.values,     
        transform = get_test_transforms()
    )
    
    # data loader is used to load data
    test_loader = DataLoader(
        test_dataset, batch_size=params['batch_size'],
        shuffle=False, num_workers=params['num_workers'],
        pin_memory=True
    )

    temp_preds = None
    with torch.no_grad():
        for images in test_loader:
            images = images.to(params['device'], non_blocking=True)
            predictions = model(images).softmax(dim=1).argmax(dim=1).to('cpu').numpy()      # apply softmax and find the highest possibility
            
            if temp_preds is None:
                temp_preds = predictions
            else:
                temp_preds = np.hstack((temp_preds, predictions))

    test_df[f'model_{i}_preds'] = temp_preds
    pred_cols.append(f'model_{i}_preds')

test_df['label'] = test_df[pred_cols].mode(axis=1)[0]
test_df = test_df[['image_id', 'label']]
test_df['label'] = test_df['label'].map(id2label)

test_df.to_csv('./output/submission.csv', index=False)



