"""
Multi-task Dataset for Benign/Malignant Classification and BI-RADS Classification
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


DEFAULT_BIRADS_ORDER = ['3', '4A', '4B', '4C', '5']


def normalize_birads_label(birads_value):
    """Normalize BI-RADS labels to a stable uppercase string form."""
    if pd.isna(birads_value):
        return None

    birads_str = str(birads_value).upper().strip()
    return birads_str or None


def build_birads_mapping(values=None, base_order=None):
    """
    Build a stable BI-RADS mapping.

    Known BI-RADS classes keep their clinical order so train/val/eval share
    the same label semantics even when a split is missing one class.
    """
    base_order = list(base_order or DEFAULT_BIRADS_ORDER)
    seen = set(base_order)

    extras = []
    for value in values or []:
        normalized = normalize_birads_label(value)
        if normalized is None or normalized in seen:
            continue
        extras.append(normalized)
        seen.add(normalized)

    ordered_labels = base_order + sorted(extras)
    return {label: idx for idx, label in enumerate(ordered_labels)}


class MultiTaskDataset(Dataset):
    """
    Dataset for multi-task learning:
    - Task 1: Benign/Malignant classification (binary: 0=benign, 1=malignant)
    - Task 2: BI-RADS classification (6 classes: 3, 4A, 4B, 4C, 5, and possibly others)
    """
    
    def __init__(self, label_csv, image_dir, transform=None, mode='train', birads_mapping=None):
        """
        Args:
            label_csv: Path to label.csv file
            image_dir: Root directory containing images
            transform: Optional transform to be applied on images
            mode: 'train' or 'val'
        """
        self.label_df = pd.read_csv(label_csv)
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        
        # Normalize BI-RADS values once so a stable mapping can be reused.
        self.label_df['BI-RADS_normalized'] = self.label_df['BI-RADS'].apply(normalize_birads_label)

        # Map BI-RADS categories to class indices
        if birads_mapping is None:
            self.birads_mapping = build_birads_mapping(self.label_df['BI-RADS_normalized'].tolist())
        else:
            self.birads_mapping = {
                normalize_birads_label(label): int(idx)
                for label, idx in birads_mapping.items()
            }
        self.num_birads_classes = len(self.birads_mapping)
        
        # Process BI-RADS labels
        self.label_df['birads_class'] = self.label_df['BI-RADS_normalized'].apply(
            lambda x: self._process_birads(x)
        )
        
        # Remove samples with invalid BI-RADS labels
        self.label_df = self.label_df[self.label_df['birads_class'] != -1]
        
        print(f"[{mode}] Loaded {len(self.label_df)} samples")
        print(f"[{mode}] BI-RADS classes: {self.birads_mapping}")
        print(f"[{mode}] Label distribution - Benign: {(self.label_df['label']==0).sum()}, "
              f"Malignant: {(self.label_df['label']==1).sum()}")
        
        # Print BI-RADS distribution
        birads_dist = self.label_df['birads_class'].value_counts().sort_index()
        print(f"[{mode}] BI-RADS distribution:")
        for idx, count in birads_dist.items():
            birads_name = [k for k, v in self.birads_mapping.items() if v == idx][0]
            print(f"  {birads_name}: {count}")
    
    def _process_birads(self, birads_value):
        """Process BI-RADS value and return class index"""
        if birads_value is None:
            return -1

        return self.birads_mapping.get(birads_value, -1)
    
    def __len__(self):
        return len(self.label_df)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Transformed image tensor
            label_benign_malignant: Binary label (0=benign, 1=malignant)
            label_birads: BI-RADS class index
        """
        row = self.label_df.iloc[idx]
        img_id = row['id']
        label = row['label']
        
        # Find image file (try multiple locations)
        img_path = None
        
        # Location 1: Direct in image_dir
        for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
            potential_path = os.path.join(self.image_dir, f"{img_id}{ext}")
            if os.path.exists(potential_path):
                img_path = potential_path
                break
        
        # Location 2: In subdirectory by label (0/ or 1/)
        if img_path is None:
            for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
                potential_path = os.path.join(self.image_dir, str(label), f"{img_id}{ext}")
                if os.path.exists(potential_path):
                    img_path = potential_path
                    break
        
        if img_path is None:
            raise FileNotFoundError(f"Image not found for ID: {img_id} in {self.image_dir} or {self.image_dir}/{label}/")
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        # Get labels
        label_benign_malignant = torch.tensor(row['label'], dtype=torch.long)
        label_birads = torch.tensor(row['birads_class'], dtype=torch.long)
        
        return image, label_benign_malignant, label_birads, img_id


def get_transforms(image_size=224, mode='train'):
    """Get data transforms for training or validation"""
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
