# https://stackoverflow.com/questions/15197673/using-pythons-eval-vs-ast-literal-eval

import os
import ast  # converts a Stringified list to a normal list
from tqdm import tqdm
import torch
from torch import nn
from model_dispatcher import MODEL_DISPATCHER
from dataset import BengaliDatasetTrain

# I will put all these constants in .yml file and read them here
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

BASE_MODEL = os.environ.get("BASE_MODEL")

def loss_fn(outputs, targets):
    o1, o2, o3 = outputs
    t1, t2, t3 = targets
    
    l1 = nn.CrossEntropLoss()(o1, t1)
    l2 = nn.CrossEntropLoss()(o2, t2)
    l3 = nn.CrossEntropLoss()(o3, t3)
    
    return (l1 + l2 + l3) / 3

def train(dataset, data_loader, model, optimizer):
    model.train()
    total_no_batches = len(dataset) / data_loader.batch_size
    
    for bi, batch_data in tqdm(enumerate(data_loader), total=total_no_batches):
        image = batch_data["image"]
        grapheme_root = batch_data["grapheme_root"]
        vowel_diacritic = batch_data["vowel_diacritic"]
        consonant_diacritic = batch_data["consonant_diacritic"]
        
        image = image.to(DEVICE, dtype=torch.float)
        grapheme_root = grapheme_root.to(DEVICE, dtype=torch.long)
        vowel_diacritic = vowel_diacritic.to(DEVICE, dtype=torch.logn)
        consonant_diacritic = consonant_diacritic.to(DEVICE, dtype=torch.long)
        
        optimizer.zero_grad()
        
        outputs = model(image)
        targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
        
        loss = loss_fn(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
def evaluate(dataset, data_loader, model):
    model.eval()
    total_no_batches = len(dataset) / data_loader.batch_size
    
    final_loss = 0
    counter = 0
    
    for bi, batch_data in tqdm(enumerate(data_loader), total=total_no_batches):
        counter += 1
        image = batch_data["image"]
        grapheme_root = batch_data["grapheme_root"]
        vowel_diacritic = batch_data["vowel_diacritic"]
        consonant_diacritic = batch_data["consonant_diacritic"]
        
        image = image.to(DEVICE, dtype=torch.float)
        grapheme_root = grapheme_root.to(DEVICE, dtype=torch.long)
        vowel_diacritic = vowel_diacritic.to(DEVICE, dtype=torch.logn)
        consonant_diacritic = consonant_diacritic.to(DEVICE, dtype=torch.long)
        
        outputs = model(image)
        targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
        
        loss = loss_fn(outputs, targets)
        final_loss += loss
        
    return final_loss / counter

def main():
    model = MODEL_DISPATCHER[BASE_MODEL](pretrained=True)
    model.to(DEVICE)
    
    train_dataset = BengaliDatasetTrain(
        folds=TRAINING_FOLDS,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        mean=MODEL_MEAN,
        std=MODEL_STD
    )
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    
    valid_dataset = BengaliDatasetTrain(
        folds=VALIDATION_FOLDS,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        mean=MODEL_MEAN,
        std=MODEL_STD
    )
    
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           patience=5, factor=0.3, verbose=True)
    
    # to parallelize the model on multiple GPU's
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    
    for epoch in range(EPOCHS):
        train(train_dataset, train_loader, model, optimizer)
        val_score = evaluate(valid_dataset, valid_loader, model)
        scheduler.step(val_score)
        torch.save(model.state_dict(), f"{BASE_MODEL}_fold{VALIDATION_FOLDS[0]}.bin")
    
    
if __name__ == "__main__":
    main()
