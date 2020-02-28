# if we read the images using dataframe all the time, it will be very slow
# so we will convert each image into pickle format file to speed up the work

import pandas as pd
import joblib
import glob
from tqdm import tqdm

if __name__ == "__main__":
    files = glob.glob("../input/bengaliai-cv19/train_*.parquet")
    for file in files:
        df = pd.read_parquet(file)
        image_ids = df.image_id.values
        df = df.drop("image_id", axis=1)
        image_arrays = df.values
        for img_idx, img_id in tqdm(enumerate(image_ids), total=len(image_ids)):
            joblib.dump(image_arrays[img_idx, :], f"../input/image_pickles/{img_id}.pkl")
