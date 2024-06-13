from pathlib import Path

KEYWORD_EXTRACTION_MODEL = "ml6team/keyphrase-extraction-distilbert-inspec"

# MIMIC-CXR constants
MIMIC_H5_PATH = Path("/mnt/data/MIMIC-CXR-JPG")
MIMIC_CXR_DATA_DIR = Path("/mnt/data/MIMIC-CXR/physionet.org/files/mimic-cxr/2.0.0/")
MIMIC_CXR_JPG_DATA_DIR = Path("/mnt/data/MIMIC-CXR-JPG/physionet.org/files/mimic-cxr-jpg/2.1.0")

MIMIC_CXR_SPLIT_CSV = MIMIC_CXR_JPG_DATA_DIR / "mimic-cxr-2.0.0-split-clean.csv.gz"
MIMIC_CXR_LABELS_CSV = MIMIC_CXR_JPG_DATA_DIR / "mimic-cxr-2.0.0-chexpert.csv.gz"
MIMIC_CXR_METADATA_CSV = MIMIC_CXR_JPG_DATA_DIR / "mimic-cxr-2.0.0-metadata.csv.gz"
