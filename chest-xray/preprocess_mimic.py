import pandas as pd
from constants import *
from PIL import Image
from tqdm import tqdm
import numpy as np
import h5py
import re

IMG_SIZE = 224


def prepare_data():
    """Load and merge dataframes, filter rows, and drop unnecessary columns."""
    img_df = pd.read_csv(MIMIC_CXR_SPLIT_CSV)
    label_df = pd.read_csv(MIMIC_CXR_LABELS_CSV).fillna(0).replace(-1, 0)
    metadata_df = pd.read_csv(MIMIC_CXR_METADATA_CSV)[['dicom_id', 'subject_id', 'study_id', 'ViewPosition']]

    merged_df = pd.merge(img_df, label_df, on=['subject_id', 'study_id'])
    merged_df = pd.merge(merged_df, metadata_df, on=['dicom_id', 'subject_id', 'study_id'])
    filtered_df = merged_df[merged_df['ViewPosition'].isin(['PA', 'AP'])]

    return filtered_df.drop(columns=['ViewPosition'])


def process_image(image_path):
    """Resize image to 224x224 while maintaining aspect ratio and convert to grayscale."""
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    original_size = image.size
    ratio = min(IMG_SIZE / original_size[0], IMG_SIZE / original_size[1])
    new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
    resized_image = image.resize(new_size, Image.LANCZOS)

    # Create a new image with padding
    new_image = Image.new("L", (IMG_SIZE, IMG_SIZE))  # Grayscale mode
    new_image.paste(resized_image, ((IMG_SIZE - new_size[0]) // 2, (IMG_SIZE - new_size[1]) // 2))

    return np.array(new_image)


def extract_sections(report_content):
    """Extract 'Findings' and 'Impression' sections from the report."""
    findings_match = re.search(r'FINDINGS:(.*?)(?:IMPRESSION:|$)', report_content, re.DOTALL)
    impression_match = re.search(r'IMPRESSION:(.*)', report_content, re.DOTALL)
    
    # Extract the findings and impression texts
    findings_text = findings_match.group(1).strip() if findings_match else ''
    impression_text = impression_match.group(1).strip() if impression_match else ''

    return findings_text, impression_text

def preprocess_report(report_content):
    """Keep only 'Findings' and 'Impression' sections and filter out invalid reports."""
    findings_text, impression_text = extract_sections(report_content)

    # Check if both sections are empty or contain less than two words
    if (not findings_text and not impression_text) or \
       (len(findings_text.split()) < 2 and len(impression_text.split()) < 2):
        return None
    
    # Prepare the final report format
    processed_report = f"FINDINGS: {findings_text}\nIMPRESSION: {impression_text}"
    return processed_report


def save_split_to_h5(split_df, split, h5_path):
    """Save a split of the dataset to an H5 file."""
    split_path = f'{h5_path}/{split}_{IMG_SIZE}.h5'
    print(f"Processing {split} split with {len(split_df)} images")
    print(f"Saving to {split_path}")
    with h5py.File(split_path, 'w') as hdf5_file:
        # Create datasets for reports, images and labels
        reports_dataset = hdf5_file.create_dataset("reports", (len(split_df),), dtype=h5py.string_dtype(), chunks=True)
        images_dataset = hdf5_file.create_dataset("images", (len(split_df), IMG_SIZE, IMG_SIZE), dtype='uint8', chunks=True)
        labels_dataset = hdf5_file.create_dataset("labels", (len(split_df), 14), dtype='uint8', chunks=True)
    
        valid_index = 0
        for index, row in tqdm(split_df.iterrows(), total=len(split_df)):
            study_id = f"s{row['study_id']}"
            subject_id = f"p{row['subject_id']}"
            folder_name = subject_id[:3]

            # Load and extract only Findings and Impressions sections from the report
            report_path = MIMIC_CXR_DATA_DIR / "files" / folder_name / subject_id / f"{study_id}.txt"
            with open(report_path, 'r') as report_file:
                report_content = report_file.read()
            processed_report = preprocess_report(report_content)
            # If the report is lacking Findings and Impressions or is too short, skip this image
            if processed_report is None:
                continue
            reports_dataset[valid_index] = processed_report

            # Load and preprocess the image
            img_path = MIMIC_CXR_JPG_DATA_DIR / "files" / folder_name / subject_id / study_id /  f"{row['dicom_id']}.jpg"
            processed_image = process_image(img_path)
            images_dataset[valid_index] = processed_image

            label = row.drop(['subject_id', 'study_id', 'dicom_id', 'split']).astype(int).values
            labels_dataset[valid_index] = label

            valid_index += 1
        
        images_dataset.resize((valid_index, IMG_SIZE, IMG_SIZE))
        labels_dataset.resize((valid_index, 14))
        reports_dataset.resize((valid_index,))


if __name__ == "__main__":
    df = prepare_data()

    for split in ["train", "validate", "test"]:
        split_df = df[df['split'] == split]
        save_split_to_h5(split_df, split, MIMIC_H5_PATH)
