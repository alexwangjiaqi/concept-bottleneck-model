"""
General utils for training, evaluation and data loading
"""
import os
import torch
import pickle
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
N_ATTRIBUTES = 200

from torch.utils.data import BatchSampler
from torch.utils.data import Dataset, DataLoader


class CUBDataset(Dataset):
    """
    Returns a compatible Torch Dataset object customized for the CUB dataset
    """

    def __init__(self, pkl_file_paths, use_attr=True, no_img=False, uncertain_label=False, transform=None):
        """
        Args:
            pkl_file_paths (list): List of full paths to .pkl data files.
            use_attr (bool): Whether to load and return attribute (concept) labels.
                            Set to False when only using class labels (e.g., for X → Y training).
            no_img (bool): Whether to skip loading images. Set to True for concept-only input (e.g., C → Y).
            uncertain_label (bool): If True, use soft concept labels from 'uncertain_attribute_label';
                                    otherwise use hard binary labels from 'attribute_label'.
            transform (callable, optional): Optional torchvision transform to apply to the image.
        """
        self.data = []
        for file_path in pkl_file_paths:
            self.data.extend(pickle.load(open(file_path, 'rb')))
        self.transform = transform
        self.use_attr = use_attr
        self.no_img = no_img
        self.uncertain_label = uncertain_label

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        sample = self.data[idx]
        img_path = sample['img_path']

        # Load image unless no_img is True
        if not self.no_img:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)

        # Get concept labels
        if self.use_attr:
            attr_label = sample['uncertain_attribute_label'] if self.uncertain_label else sample['attribute_label']
            attr_label = torch.tensor(attr_label, dtype=torch.float32)

        # Get class label
        class_label = torch.tensor(sample['class_label'], dtype=torch.long)

        # Return appropriate tuple
        if self.no_img and self.use_attr:
            return attr_label, class_label
        elif self.no_img and not self.use_attr:
            return class_label
        elif not self.no_img and self.use_attr:
            return img, attr_label, class_label
        else:
            return img, class_label



def load_data(pkl_paths, use_attr=True, no_img=False, batch_size=32, 
              uncertain_label=True, image_dir='images', transform=None):
    """
    Loads a DataLoader from given pickle file paths.

    Args:
        pkl_paths (list of str): paths to train/val/test .pkl files
        use_attr (bool): whether to load concept labels
        no_img (bool): whether to return just concepts (e.g., for C → Y training)
        batch_size (int): batch size
        uncertain_label (bool): use soft labels (weighted by attribute certainty)
        image_dir (str): base folder for image loading
        transform (callable): image transformation pipeline (optional)

    Returns:
        DataLoader: PyTorch DataLoader for training or evaluation
    """

    is_training = any(['train.pkl' in path for path in pkl_paths])

    # Define default transforms if none given
    if transform is None:
        if is_training:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225]),
            ])

    dataset = CUBDataset(
        pkl_file_paths=pkl_paths,
        use_attr=use_attr,
        no_img=no_img,
        uncertain_label=uncertain_label,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        drop_last=is_training
    )

    return loader