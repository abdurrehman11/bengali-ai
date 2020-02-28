import os
import ast
# https://stackoverflow.com/questions/15197673/using-pythons-eval-vs-ast-literal-eval

DEVICE = "cuda"

TRAINING_FOLDS_CSV = os.environ.get("TRAINING_FOLDS_CSV")
EPOCHS = int(os.environ.get("EPOCHS"))

IMG_HEIGHT = int(os.environ.get("IMG_HEIGHT"))
IMG_WIDTH = int(os.environ.get("IMG_WIDTH"))

TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE"))
TEST_BATCH_SIZE = int(os.environ.get("TEST_BATCH_SIZE"))

MODEL_MEAN = ast.literal_eval(os.environ.get("MODEL_MEAN"))
MODEL_STD = ast.literal_eval(os.environ.get("MODEL_STD"))

TRAINING_FOLDS = ast.literal_eval(os.environ.get("TRAINING_FOLDS"))
VALIDATION_FOLDS = ast.literal_eval(os.environ.get("VALIDATION_FOLDS"))

BASE_MODEL = int(os.environ.get("BASE_MODEL"))
