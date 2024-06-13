import pandas as pd
from constants import *
from PIL import Image
import numpy as np

df = pd.read_csv(MIMIC_CXR_SPLIT_CSV)
failed_images = []

for index, row in df.iterrows():
    if index % 1000 == 0:
        print(f"Processing image {index} / {len(df)}")
    study_id = f"s{row['study_id']}"
    subject_id = f"p{row['subject_id']}"
    folder_name = subject_id[:3]

    img_path = MIMIC_CXR_JPG_DATA_DIR / "files" / folder_name / subject_id / study_id /  f"{row['dicom_id']}.jpg"
    try:
        image = Image.open(img_path).convert('L')
    except Exception as e:
        print(f"Failed to open image {img_path}")
        print(e)
        failed_images.append(img_path)

def extract_relative_path(full_path):
    return str(full_path.relative_to('/mnt/data/MIMIC-CXR-JPG/physionet.org/files/mimic-cxr-jpg/2.1.0'))

# Create a list of relative paths
relative_paths = [extract_relative_path(path) for path in failed_images]

# Write the relative paths to a file
with open('IMAGE_FILENAMES', 'w') as file:
    for path in relative_paths:
        file.write(f"{path}\n")