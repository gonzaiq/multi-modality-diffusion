import os
from torch.utils.data import Dataset
from PIL import Image
import torch
import torchvision.transforms as transforms
import re

class CityscapesDataset(Dataset):
    def __init__(self, image_dir, mask_dir, split, transform_image=None, transform_mask=None):
        """
        Args:
            data_dir (str): Path to the directory containing images.
            transform (callable, optional): Optional transform to be applied to an image.
        """
        self.image_dir = os.path.join(image_dir, split)
        self.mask_dir = os.path.join(mask_dir, split)
        self.elements = []
        for city in os.listdir(self.image_dir):
            _img_subdir = os.path.join(self.image_dir, city)
            _msk_subdir = os.path.join(self.mask_dir, city)
            for img_pth in os.listdir(_img_subdir):
                msk_pth = re.sub(r'leftImg8bit', 'gtFine_color', img_pth)
                elem = {
                    "city": city,
                    "img_pth": os.path.join(_img_subdir, img_pth),
                    "msk_pth": os.path.join(_msk_subdir, msk_pth)
                }
                self.elements.append(elem)

        self.transform_image = transform_image
        self.transform_mask = transform_mask

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, idx):

        city = self.elements[idx]["city"]
        img_pth = self.elements[idx]["img_pth"]
        msk_pth = self.elements[idx]["msk_pth"]

        image = Image.open(img_pth).convert("RGB")
        mask = Image.open(msk_pth).convert("RGB")
        
        if self.transform_image:
            image = self.transform_image(image)

        if self.transform_mask:
            mask = self.transform_mask(mask)
        
        return {"city": city, "image": image, "mask": mask}