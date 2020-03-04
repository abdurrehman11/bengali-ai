# https://stackoverflow.com/questions/15197673/using-pythons-eval-vs-ast-literal-eval
import ast

from tqdm import tqdm
import torch
from torch import nn
from src.model_dispatcher import MODEL_DISPATCHER
from src.dataset import BengaliDatasetTrain
import yaml
from src.BengaliConstants import BengaliConstants


def load_config():
    with open('bengali_config.yml', 'rt') as file:
        model_config = yaml.safe_load(file.read())

    return model_config


def loss_fn(outputs, targets):
    o1, o2, o3 = outputs
    t1, t2, t3 = targets

    l1 = nn.CrossEntropyLoss()(o1, t1)
    l2 = nn.CrossEntropyLoss()(o2, t2)
    l3 = nn.CrossEntropyLoss()(o3, t3)

    return (l1 + l2 + l3) / 3


def train(dataset, data_loader, model, optimizer):
    model_config = load_config()
    model.train()
    total_no_batches = len(dataset) / data_loader.batch_size

    for bi, batch_data in tqdm(enumerate(data_loader), total=total_no_batches):
        image = batch_data["image"]
        grapheme_root = batch_data["grapheme_root"]
        vowel_diacritic = batch_data["vowel_diacritic"]
        consonant_diacritic = batch_data["consonant_diacritic"]

        device = model_config[BengaliConstants.MODEL_CONFIG][BengaliConstants.DEVICE]
        image = image.to(device, dtype=torch.float)
        grapheme_root = grapheme_root.to(device, dtype=torch.long)
        vowel_diacritic = vowel_diacritic.to(device, dtype=torch.long)
        consonant_diacritic = consonant_diacritic.to(device, dtype=torch.long)

        optimizer.zero_grad()

        outputs = model(image)
        targets = (grapheme_root, vowel_diacritic, consonant_diacritic)

        loss = loss_fn(outputs, targets)

        loss.backward()
        optimizer.step()


def evaluate(dataset, data_loader, model):
    model_config = load_config()
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

        device = model_config[BengaliConstants.MODEL_CONFIG][BengaliConstants.DEVICE]
        image = image.to(device, dtype=torch.float)
        grapheme_root = grapheme_root.to(device, dtype=torch.long)
        vowel_diacritic = vowel_diacritic.to(device, dtype=torch.long)
        consonant_diacritic = consonant_diacritic.to(device, dtype=torch.long)

        outputs = model(image)
        targets = (grapheme_root, vowel_diacritic, consonant_diacritic)

        loss = loss_fn(outputs, targets)
        final_loss += loss

    return final_loss / counter


def main():
    model_config = load_config()
    model = MODEL_DISPATCHER[model_config[BengaliConstants.MODEL_CONFIG][BengaliConstants.BASE_MODEL]](pretrained=True)
    model.to(model_config[BengaliConstants.MODEL_CONFIG][BengaliConstants.DEVICE])

    height = model_config[BengaliConstants.MODEL_CONFIG][BengaliConstants.IMG_HEIGHT]
    width = model_config[BengaliConstants.MODEL_CONFIG][BengaliConstants.IMG_WIDTH]
    model_mean = ast.literal_eval(model_config[BengaliConstants.MODEL_CONFIG][BengaliConstants.MODEL_MEAN])
    model_std = ast.literal_eval(model_config[BengaliConstants.MODEL_CONFIG][BengaliConstants.MODEL_STD])

    for fold in range(model_config[BengaliConstants.MODEL_CONFIG][BengaliConstants.FOLDS]):
        train_dataset = BengaliDatasetTrain(
            folds=model_config[BengaliConstants.MODEL_CONFIG][BengaliConstants.TRAINING_FOLDS][fold],
            img_height=height,
            img_width=width,
            mean=model_mean,
            std=model_std
        )

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=model_config[BengaliConstants.MODEL_CONFIG][BengaliConstants.TRAIN_BATCH_SIZE],
            shuffle=True,
            num_workers=4
        )

        valid_dataset = BengaliDatasetTrain(
            folds=model_config[BengaliConstants.MODEL_CONFIG][BengaliConstants.VALIDATION_FOLDS][fold],
            img_height=height,
            img_width=width,
            mean=model_mean,
            std=model_std
        )

        valid_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset,
            batch_size=model_config[BengaliConstants.MODEL_CONFIG][BengaliConstants.TEST_BATCH_SIZE],
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

        for epoch in range(model_config[BengaliConstants.MODEL_CONFIG][BengaliConstants.EPOCHS]):
            train(train_dataset, train_loader, model, optimizer)
            val_score = evaluate(valid_dataset, valid_loader, model)
            scheduler.step(val_score)

            model_name = model_config[BengaliConstants.MODEL_CONFIG][BengaliConstants.BASE_MODEL] + "_fold_" + \
                         model_config[BengaliConstants.MODEL_CONFIG][BengaliConstants.VALIDATION_FOLDS][fold][0]
            print("Model Name: ", model_name)

            torch.save(model.state_dict(), f"../models/{model_name}.bin")


if __name__ == "__main__":
    main()
