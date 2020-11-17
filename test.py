from torch.utils.data import Dataset
from aicsimageio import AICSImage, imread
import os
from PIL import Image
import numpy as np
import torch
from matplotlib import pyplot as plt
import segmentation_models_pytorch as smp
import albumentations as albu


class UniversalDataset(Dataset):
    SEMANTIC_SEGMENTAION = 0
    INSTANCE_SEGMENTAION = 1
    INSTANCE_DETECTION = 2

    POLYGON_MASK = 0
    BINARY_MASK = 1
    MULTICOLOR_MASK = 2

    def __init__(self, root, output_data_type=SEMANTIC_SEGMENTAION, input_mask_type=POLYGON_MASK):
        self.root = root
        self.output_data_type = output_data_type
        self.input_mask_type = input_mask_type
        self.multicolor_data_paths = {"Images":[],"Masks":[]}

        self.scan_root()

    def scan_root(self):
        if self.input_mask_type is self.POLYGON_MASK:
            pass
        elif self.input_mask_type is self.BINARY_MASK:
            pass
        elif self.input_mask_type is self.MULTICOLOR_MASK:
            self.multicolor_data_paths["Images"] = list(sorted(os.listdir(os.path.join(self.root, "Images"))))
            self.multicolor_data_paths["Masks"] = list(sorted(os.listdir(os.path.join(self.root, "Masks"))))

    def set_output_data_type(self, output_data_type):
        self.output_data_type = output_data_type

    def set_input_mask_type(self, input_mask_type):
        self.input_mask_type = input_mask_type
        self.scan_root()

    def set_root(self, root):
        self.root = root
        self.scan_root()

    def __len__(self):
        if self.input_mask_type is self.MULTICOLOR_MASK:
            return len(self.multicolor_data_paths["Images"])

    def __getitem__(self, idx):
        if self.output_data_type is self.SEMANTIC_SEGMENTAION:

            pass

        elif self.output_data_type is self.INSTANCE_SEGMENTAION:

            if self.input_mask_type is self.POLYGON_MASK:
                pass
            elif self.input_mask_type is self.BINARY_MASK:
                pass
            elif self.input_mask_type is self.MULTICOLOR_MASK:

                image_path = os.path.join(self.root, "Images", self.multicolor_data_paths["Images"][idx])
                mask_path = os.path.join(self.root, "Masks", self.multicolor_data_paths["Masks"][idx])
                if image_path.endswith(".czi"):
                    bio_image = AICSImage(image_path)
                    image_data = bio_image.get_image_data("ZYX", C=bio_image.get_channel_names().index("DAPIr2"), S=0, T=0)
                    image_data = image_data.mean(axis=0)
                    image = Image.fromarray(image_data).convert("RGB")
                    plt.imshow(image)
                    plt.show()
                else:
                    image = Image.open(image_path).convert("RGB")
                mask = Image.open(mask_path)
                mask = np.array(mask)
                obj_ids = np.unique(mask)
                obj_ids = obj_ids[1:]
                masks = mask == obj_ids[:, None, None]
                # num_objs = len(obj_ids)
                # boxes = []
                # for i in range(num_objs):
                #     pos = np.where(masks[i])
                #     xmin = np.min(pos[1])
                #     xmax = np.max(pos[1])
                #     ymin = np.min(pos[0])
                #     ymax = np.max(pos[0])
                #     boxes.append([xmin, ymin, xmax, ymax])
                # boxes = torch.as_tensor(boxes, dtype=torch.float32)
                # labels = torch.ones((num_objs,), dtype=torch.int64)
                masks = torch.as_tensor(masks, dtype=torch.uint8)
                target = masks
                # image_id = torch.tensor([idx])
                # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                # iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
                # target = {}
                # target["boxes"] = boxes
                # target["labels"] = labels
                # target["masks"] = masks
                # target["image_id"] = image_id
                # target["area"] = area
                # target["iscrowd"] = iscrowd

                return image, target
            
        elif self.output_data_type is self.INSTANCE_DETECTION:

            pass

def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

if __name__ == '__main__':
    dataset = UniversalDataset(
        "C:/Users/gator/Downloads/PennFudanPed/PennFudanPed", 
        output_data_type=UniversalDataset.INSTANCE_SEGMENTAION, 
        input_mask_type=UniversalDataset.MULTICOLOR_MASK
    )
    model = smp.Unet()
    model = smp.Unet('resnet34', encoder_weights='imagenet')
    model = smp.Unet('resnet34', classes=3, activation='softmax')
    image, mask = dataset[4]
    visualize(
        image=image, 
        cars_mask=mask[0].squeeze(),
    )