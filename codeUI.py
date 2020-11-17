from torch.utils.data import Dataset
import json
import os
from PIL import Image
from aicsimageio import AICSImage, imread
from matplotlib import pyplot as plt

dataset_path = "/home/techgarage/Projects/Max Planck/AIModelTrainer/datasets/default.json"
model_type = ""
model_name = ""
transformations = []


class UniversalDataset(Dataset):
    def __init__(self, dataset_path, model_class, transformations=[]):
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
        for image in dataset_dict["images"]:
            if(os.path.exists(image["file_name"])): 
                self.__image_path_list.append([image["file_name"], image["channel"]])
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
                image = image.get_image_data("ZYX", C=int(channel), S=0, T=0)
            
            image = image.mean(axis=0)
            image = Image.fromarray(image).convert("RGB")
            plt.imshow(image)
            plt.show()
        
        return image
            

    def __len__(self):
        return len(self.__image_path_list)
    
    def __getitem__(self, idx):
        image = self.load_image(self.__image_path_list[idx][0],self.__image_path_list[idx][1])
        return image

test_dataset = UniversalDataset(dataset_path, model_type, transformations=transformations)
for image in test_dataset:
    pass