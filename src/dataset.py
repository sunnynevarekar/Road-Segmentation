import os
import numpy as np
from PIL import Image
import torch

class SegDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir, ids, transforms=None):
        """Initialize segmentation dataset
           Args:
            img_dir: directory for images, str
            mask_dir: directory for masks, str
            ids: filename identifiers for image and mask file, 
                 a single sample has sam id for image and corresponding mask
            transforms: Augumentations and tranforms for images and mask
        """
        super(SegDataset, self).__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.ids = ids
        self.transforms = transforms

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.ids[idx]+'.png')
        mask_path = os.path.join(self.mask_dir, self.ids[idx]+'.png')

        img = np.array(Image.open(img_path).convert('RGB'), dtype=np.float32)/255.
        #convert mask pixel values 0, 255 to 0 and 1
        mask = np.array(Image.open(mask_path), dtype=np.float32)/255.

        if self.transforms:
            sample = self.transforms(image=img, mask=mask)
            img, mask = sample['image'], sample['mask']
            
        
        #HWC->CHW
        img = img.transpose(2, 0, 1)
        mask = np.expand_dims(mask, axis=0)
        return img.astype(np.float32), mask.astype(np.float32)

    def __len__(self):
        return len(self.ids)    

