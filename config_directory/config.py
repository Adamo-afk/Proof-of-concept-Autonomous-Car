# import the necessary packages
import os

# initialize the base path to the directory that will contain
# the images
BASE_PATH = "traffic_signs_dataset"

# define the names of the training, testing, and validation
# directories
TRAIN = "Train"
TEST = "Test"
VALIDATION = "Validation"

# initialize the list of class label names
CLASSES = ["Ahead_only", "STOP", "Turn_left_ahead", "Turn_right_ahead"]

# size of the images in pixels
IMAGE_SIZE = [32, 64]

# number of channels for each image in the dataset
IMAGE_CHANNELS = 3

# color mode of the images
COLOR_MODE = "rgb"

# set the batch size when fine-tuning
BATCH_SIZE = 25

# set the number of epochs when training the head of the model
HEAD_EPOCHS = 5

# set the number of epochs when training both the
# CONV layers along with the set of FC layers
MODEL_EPOCHS = 10

# set the path and name to the serialized model after training and retraining
MODEL_NAME = "traffic_signs.model"
MODEL_PATH = os.path.sep.join(["output", MODEL_NAME])

# define the path to the output training history plots
WARMUP_PLOT_PATH = os.path.sep.join(["output", "warmup.png"])
UNFROZEN_PLOT_PATH = os.path.sep.join(["output", "unfrozen.png"])
CLASS_DISTRIBUTION = os.path.sep.join(["output", "class_distribution.png"])
TRAIN_IMAGES_GRID = os.path.sep.join(["output", "train_grid.png"])
