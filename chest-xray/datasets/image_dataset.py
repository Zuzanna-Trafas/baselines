import cv2
import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset
from constants import *
# from models.keyword_extractor import KeyphraseExtractionPipeline


class ImageBaseDataset(Dataset):
    def __init__(
        self,
        cfg,
        split="train",
        transform=None,
    ):

        self.cfg = cfg
        self.transform = transform
        self.split = split

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def read_from_jpg(self, img_path):

        x = cv2.imread(str(img_path), 0)

        # tranform images
        if self.cfg.get("data") is not None and self.cfg.data.get("image") is not None:
            x = self._resize_img(x, self.cfg.data.image.imsize)
        img = Image.fromarray(x).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img

    def _resize_img(self, img, scale):
        """
        Args:
            img - image as numpy array (cv2)
            scale - desired output image-size as scale x scale
        Return:
            image resized to scale x scale with shortest dimension 0-padded
        """
        size = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)

        # Resizing
        if max_ind == 0:
            # image is heigher
            wpercent = scale / float(size[0])
            hsize = int((float(size[1]) * float(wpercent)))
            desireable_size = (scale, hsize)
        else:
            # image is wider
            hpercent = scale / float(size[1])
            wsize = int((float(size[0]) * float(hpercent)))
            desireable_size = (wsize, scale)
        resized_img = cv2.resize(
            img, desireable_size[::-1], interpolation=cv2.INTER_AREA
        )  # this flips the desireable_size vector

        # Padding
        if max_ind == 0:
            # height fixed at scale, pad the width
            pad_size = scale - resized_img.shape[1]
            left = int(np.floor(pad_size / 2))
            right = int(np.ceil(pad_size / 2))
            top = int(0)
            bottom = int(0)
        else:
            # width fixed at scale, pad the height
            pad_size = scale - resized_img.shape[0]
            top = int(np.floor(pad_size / 2))
            bottom = int(np.ceil(pad_size / 2))
            left = int(0)
            right = int(0)
        resized_img = np.pad(
            resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
        )

        return resized_img


class MIMICImageDataset(ImageBaseDataset):
    def __init__(self, cfg, split=None, transform=None):

        if MIMIC_CXR_DATA_DIR is None:
            raise RuntimeError(
                "MIMIC-CXR data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://physionet.org/content/mimic-cxr/2.0.0/"
                + f" and update MIMIC_CXR_DATA_DIR in constants.py"
            )
        
        if MIMIC_CXR_JPG_DATA_DIR is None:
            raise RuntimeError(
                "MIMIC-CXR-JPG data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://physionet.org/content/mimic-cxr-jpg/2.1.0/"
                + f" and update MIMIC_CXR_JPG_DATA_DIR in constants.py"
            )
        
        # self.extractor = KeyphraseExtractionPipeline(model=KEYWORD_EXTRACTION_MODEL)

        self.cfg = cfg

        # read in csv file
        self.df = pd.read_csv(MIMIC_CXR_SPLIT_CSV)

        if split is not None:
            self.df = self.df[self.df['split'] == split]

        self.label_df = pd.read_csv(MIMIC_CXR_LABELS_CSV)

        # sample data
        if cfg.data.get('frac') is not None and cfg.data.frac != 1 and split == "train":
            self.df = self.df.sample(frac=cfg.data.frac)

        super(MIMICImageDataset, self).__init__(cfg, split, transform)

    def preprocess_text(self, text):
        # TODO
        return text
    
    # def extract_keywords(self, text):
    #     keywords = self.extractor(text)
    #     return keywords

    def preprocess_label(self, label):
        label = label.drop(['subject_id', 'study_id'], axis=1).iloc[0] #.to_dict()
        label = label.fillna(0).replace(-1, 0)
        return label.values

    def __getitem__(self, index):
        row = self.df.iloc[index]

        study_id = f"s{row['study_id']}"
        subject_id = f"p{row['subject_id']}"
        folder_name = subject_id[:3]

        img_path = MIMIC_CXR_JPG_DATA_DIR / "files" / folder_name / subject_id / study_id /  f"{row['dicom_id']}.jpg"
        report_path = MIMIC_CXR_DATA_DIR / "files" / folder_name / subject_id / f"{study_id}.txt"

        # Load image
        image = self.read_from_jpg(img_path)

        # Load report
        with open(report_path, 'r') as file:
            report = file.read()

        report = self.preprocess_text(report)

        # keywords = self.extract_keywords(report)

        # Load label
        label_row = self.label_df[self.label_df['study_id'] == row['study_id']]
        label = self.preprocess_label(label_row)

        return {
            'image': image,
            'image_path': str(img_path),
            'report': report,
            # 'keywords': keywords,
            'label': label
        }

    def __len__(self):
        return len(self.df)