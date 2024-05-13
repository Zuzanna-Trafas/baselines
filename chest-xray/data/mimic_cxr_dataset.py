import pandas as pd
import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from constants import MIMIC_CXR_DIR, MIMIC_CXR_JPG_DIR, LABELS_CSV, SPLIT_CSV

class MIMIC_CXR_Dataset(Dataset):
    """
    Custom dataset class for the MIMIC-CXR dataset.

    Args:
        split (str, optional): One of 'train', 'validate', or 'test'. Defaults to None.
        transform (callable, optional): A function/transform to apply to the image. Defaults to None.

    Returns:
        tuple: A tuple containing the following elements:
            - image (numpy.ndarray): The image data as a NumPy array.
            - image_path (str): The path to the image file.
            - report (str): The text content of the radiology report.
            - label (dict): A dictionary containing labels associated with the study.
    """
    def __init__(self, split=None, transform=None):
        # Load metadata CSV files
        labels_path = os.path.join(MIMIC_CXR_JPG_DIR, LABELS_CSV)
        self.labels_df = pd.read_csv(labels_path)
        split_path = os.path.join(MIMIC_CXR_JPG_DIR, SPLIT_CSV)
        self.split_df = pd.read_csv(split_path)
        self.split_df = self._split(split)

        self.image_dir = MIMIC_CXR_JPG_DIR
        self.report_dir = MIMIC_CXR_DIR
        self.transform = transform

    def _split(self, split):
        if split is None:
            return self.metadata_df
        else:
            return self.split_df[self.split_df['split'] == split]

    def __len__(self):
        return len(self.split_df)

    def __getitem__(self, idx):
        row = self.split_df.iloc[idx]
        study_id = f"s{row['study_id']}"
        subject_id = f"p{row['subject_id']}"
        folder_name = subject_id[:3]

        image_filename = row['dicom_id'] + '.jpg'
        image_path = os.path.join(self.image_dir, 'files', folder_name, subject_id, study_id, image_filename)

        report_filename = str(study_id) + '.txt'
        report_path = os.path.join(self.report_dir, 'files', folder_name, subject_id, report_filename)

        # Load image
        image = Image.open(image_path)
        image = np.array(image)

        if self.transform:
            image = self.transform(image)

        # Load report
        with open(report_path, 'r') as file:
            report = file.read()

        # Load label
        label_row = self.labels_df[self.labels_df['study_id'] == row['study_id']]
        label = label_row.drop(['subject_id', 'study_id'], axis=1).iloc[0].to_dict()

        return image, image_path, report, label
