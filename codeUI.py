from torch.utils.data import Dataset
import json
import os
from PIL import Image
from aicsimageio import AICSImage, imread
from matplotlib import pyplot as plt
import cv2
import numpy as np
import torch
from time import monotonic
import albumentations as A
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from engine import train_one_epoch, evaluate
import utils


class UniversalDataset(Dataset):
    def __init__(self, dataset_path, model_class, transformations=None):
        self.__dataset_path = dataset_path
        self.__model_class = model_class
        self.__transformations = transformations
        self.__dataset_dict = {}
        self.__image_path_list = []

        self.load_dataset(self.__dataset_path)
        self.verify_dataset(self.__dataset_dict)

    def load_dataset(self, dataset_path):
        with open(dataset_path) as dataset:
            self.__dataset_dict = json.load(dataset)
    
    def verify_dataset(self, dataset_dict):
        annotation_dict_temp = {}
        for annotation in dataset_dict["annotations"]:
            try:
                annotation_dict_temp[str(annotation["image_id"])].append(annotation)
            except:
                annotation_dict_temp[str(annotation["image_id"])] = []
                annotation_dict_temp[str(annotation["image_id"])].append(annotation)
        for image in dataset_dict["images"]:
            if(os.path.exists(image["file_name"])): 
                self.__image_path_list.append([image["file_name"], image["channel"], annotation_dict_temp[str(image["id"])]])
            else:
                print(image["file_name"] + " is invalid")
        print("Using {0}/{1} images".format(len(self), len(dataset_dict["images"])))

    def load_image(self, image_path, channel=0):
        try:
            image = Image.open(image_path).convert("RGB")
        except:
            image = AICSImage(image_path)
            try:
                image = image.get_image_data("ZYX", C=image.get_channel_names().index(channel), S=0, T=0)
            except:
                try:
                    image = image.get_image_data("ZYX", C=int(channel), S=0, T=0)
                except:
                    print(image_path + " could not be opened")
                    return None
            
            image = image.mean(axis=0)
            image = Image.fromarray(image).convert("RGB")
        
        return image
            

    def __len__(self):
        return len(self.__image_path_list)
    
    def __getitem__(self, idx):
        image = self.load_image(self.__image_path_list[idx][0],self.__image_path_list[idx][1])
        annotations = self.__image_path_list[idx][2]

        target = {}
        target["boxes"] = []
        target["labels"] = []
        target["masks"] = []
        target["iscrowd"] = []
        masks = []
        for annotation in annotations:
            mask = np.zeros(image.size, dtype=np.uint8)
            for polygon in annotation["segmentation"]:
                y = polygon[1::2]
                x = polygon[0::2]
                contours = np.array(list(zip(x,y)), dtype=np.int32)
                cv2.fillPoly(mask, pts=[contours], color=(255))
            masks.append(mask)

        transformed = self.__transformations(image=np.array(image, dtype=np.uint8), masks=masks)
        for annotation, mask in list(zip(annotations, transformed["masks"])):
            annotation["segmentation"] = mask

        for annotation in annotations:
            rows = np.any(annotation["segmentation"], axis=1)
            cols = np.any(annotation["segmentation"], axis=0)
            try:
                x_min, x_max = np.where(rows)[0][[0, -1]]
                y_min, y_max = np.where(cols)[0][[0, -1]]
                target["labels"].append(annotation["category_id"])
                target["iscrowd"].append(annotation["iscrowd"])
                target["masks"].append(annotation["segmentation"])
                target["boxes"].append([x_min, y_min,x_max,y_max])
            except:
                pass
        
        try:
            target["boxes"] = torch.as_tensor(target["boxes"], dtype=torch.float32)
            target["labels"] = torch.as_tensor(target["labels"], dtype=torch.int64)
            target["masks"] = torch.as_tensor(np.array(target["masks"], dtype=np.uint8), dtype=torch.uint8)
            target["image_id"] = torch.tensor([idx])
            target["iscrowd"] = torch.as_tensor(target["iscrowd"], dtype=torch.int64)
            target["area"] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (target["boxes"][:, 2] - target["boxes"][:, 0])
        except:
            target["image_id"] = torch.tensor([idx])
            target["area"] = torch.as_tensor([], dtype=torch.float32)
            target["masks"] = torch.as_tensor([], dtype=torch.uint8)
        print(target)
        return torch.as_tensor(np.transpose(transformed["image"], (2, 0, 1)), dtype=torch.float32), target

dataset_path = "/home/techgarage/Projects/Max Planck/AIModelTrainer/datasets/default.json"
model_type = ""
model_name = ""
transformations = A.Compose([
    A.RandomCrop(width=1024, height=1024),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])

test_dataset = UniversalDataset(dataset_path, model_type, transformations=transformations)

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 1
# use our dataset and defined transformations


# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

# get the model using our helper function
model = get_model_instance_segmentation(num_classes)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)

# let's train it for 10 epochs
num_epochs = 10

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader, device=device)