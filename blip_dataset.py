import time
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import requests


class BLIPDataset(Dataset):
    """
    Custom Dataset class to be sed in combination with BLIP model
    """

    def __init__(self, data_path: str = 'data_captions.csv',
                 img_dir: str = 'images-resized',
                 preprocess=None,
                 online: bool = True,
                 caption_column: str = 'caption'):
        """
        @param: data_path: string value that indicates the filename of the base data
        @param: img_dir: string value that indicates the whole folder in which there are all the images
        @param: preprocess: preprocess stage for the single image
        @param: online: boolean value that indicates if the image has to be loaded from the file system or from the web
        @param: caption_column: string value that indicates the right column of the dataset in which the caption is stored
        """
        self.img_root = img_dir
        self.data = pd.read_csv(data_path, index_col=0)
        self.preprocess = preprocess
        self.image_column = 'image_url' if online else 'name'
        self.online = online
        self.caption_column = caption_column

    def __getitem__(self, item: int):
        """
        Returns a tuple of image, [caption]
        @param: item: int value that indicates which item of the dataset must be returned
        """
        raw = self.data.iloc[item]
        name, caption = raw[[self.image_column, self.caption_column]]
        try:
            image = Image.open(requests.get(name, stream=True).raw).convert('RGB') if self.online\
                else Image.open(f'{self.img_root}/{name}').convert('RGB')
        except:
            print(f'exception for artwork at link:{name}')
            time.sleep(60)
            return self.__getitem__(item)
        return self.preprocess(image), [caption]

    def __len__(self):
        """
        Returns how many items are in the dataset       
        """
        return len(self.data)