NUM_FEAT = 31
SEQ_LENGTH = 10
FRAME_TH = 7
DIST_TH = 0.07

DATASET_DIR = "path/to/dataset"
WEIGHTS_DIR = "/path/to/model/weights"
# Folder with files, which were created during execution
VAR_DIR = "path/to/var"

classes = ["walking", "boxing", "running", "handclapping", "handwaving", "jogging"]
LABEL_ENCODER = {label: idx for idx, label in enumerate(classes)}
LABEL_DECODER = {idx: label for idx, label in enumerate(classes)}
NUM_FEAT = 31
SEQ_LENGTH = 10
FRAME_TH = 7
DIST_TH = 0.07
