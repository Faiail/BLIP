import time
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import requests

"""
TODO: Remember to calculate perplexity to check for captions quality. See https://medium.com/@priyankads/perplexity-of-language-models-41160427ed72
"""


class BLIPDataset(Dataset):
    def __init__(self, data_path: str = 'data_captions.csv',
                 img_dir: str = 'images-resized',
                 preprocess=None,
                 online: bool = True):
        self.img_root = img_dir
        self.data = pd.read_csv(data_path, index_col=0)
        self.preprocess = preprocess
        self.image_column = 'image_url' if online else 'name'
        self.online = online

    def __getitem__(self, item):
        raw = self.data.iloc[item]
        name, caption = raw[[self.image_column, 'caption']]
        try:
            image = Image.open(requests.get(name, stream=True).raw).convert('RGB') if self.online\
                else Image.open(f'{self.img_root}/{name}').convert('RGB')
        except:
            print(f'exception for artwork at link:{name}')
            time.sleep(60)
            return self.__getitem__(item)
        return self.preprocess(image), [caption]

    def __len__(self):
        return len(self.data)