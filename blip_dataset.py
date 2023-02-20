from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class BLIPDataset(Dataset):
    def __init__(self, data_path: str = 'data_captions.csv', img_dir: str = 'images-resized', preprocess=None):
        self.img_root = img_dir
        self.data = pd.read_csv(data_path, index_col=0)
        self.preprocess = preprocess

    def __getitem__(self, item):
        name, _, caption = self.data.iloc[item]
        image = Image.open(f'{self.img_root}/{name}').convert('RGB')
        return self.preprocess(image), [caption]

    def __len__(self):
        return len(self.data)