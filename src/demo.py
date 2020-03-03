import ast

import yaml
from src.BengaliConstants import BengaliConstants

model_config = None

with open('bengali_config.yml', 'rt') as file:
    model_config = yaml.safe_load(file.read())


# print(BengaliConstants.TRAIN_FOLDS_CSV_FILE)
# print(model_config[BengaliConstants.MODEL_CONFIG][BengaliConstants.TRAINING_FOLDS][0])
print(type(model_config[BengaliConstants.MODEL_CONFIG][BengaliConstants.MODEL_MEAN]))

print(type(ast.literal_eval(model_config[BengaliConstants.MODEL_CONFIG][BengaliConstants.MODEL_MEAN])))
print(type(model_config[BengaliConstants.MODEL_CONFIG][BengaliConstants.IMG_HEIGHT]))
