import pandas as pd
import cv2
from PIL import Image

from constants import *

def preprocess(img, desired_size=224):
    old_size = img.size
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    img = img.resize(new_size, Image.LANCZOS)
    # create a new image and paste the resized on it

    new_img = Image.new('L', (desired_size, desired_size))
    new_img.paste(img, ((desired_size-new_size[0])//2,
                        (desired_size-new_size[1])//2))
    return new_img

if __name__ == "__main__":
    df = pd.read_csv(MIMIC_CXR_SPLIT_CSV)
    failed_images = []
    size = 224
    for index in range(len(df)):
        row = df.iloc[index]

        study_id = f"s{row['study_id']}"
        subject_id = f"p{row['subject_id']}"
        folder_name = subject_id[:3]

        img_path = MIMIC_CXR_JPG_DATA_DIR / "files" / folder_name / subject_id / study_id /  f"{row['dicom_id']}.jpg"
        
        try:
            # read image using cv2
            img = cv2.imread(str(img_path))
            # convert to PIL Image object
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            # preprocess
            img = preprocess(img_pil, size)  

            save_path = MIMIC_CXR_JPG_DATA_DIR / "files" / folder_name / subject_id / study_id /  f"{row['dicom_id']}_{size}.jpg"
            # img.save(save_path)
            img.save(f"image_{index}.jpg")

        except Exception as e:
            failed_images.append((img_path, e))
    
    if failed_images:
        failed_images_df = pd.DataFrame(failed_images, columns=['img_path', 'error'])
        failed_images_df.to_csv("failed_images.csv", index=False)