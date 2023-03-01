from blip_dataset import BLIPDataset
import torch
from models.blip_itm import blip_itm
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader
import numpy as np
import random
from tqdm import tqdm
import json
import pandas as pd

image_size = 224  # default a 480 (fai check per vedere se è meglio)
seed = 1

print('setting seed')
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


if __name__ == "__main__":
    print('loading model')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth'
    model = blip_itm(pretrained=model_url, image_size=image_size, vit='base').to(device)

    print('creating dataset')
    transform = Compose([
        Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

    dataset = BLIPDataset(data_path='artgraph/data/data_captions.csv',
                          img_dir='artgraph/data/images-resized',
                          preprocess=transform,
                          online=False,
                          caption_column='caption')
    loader = DataLoader(dataset=dataset, batch_size=8, shuffle=False)

    print('performing evaluation')

    sum_itm = 0.0
    sum_itc = 0.0
    itm = torch.Tensor()
    itc = torch.Tensor()
    for images, captions in tqdm(loader):
        images = images.to(device)
        captions = captions[0]
        with torch.no_grad():
            itm_output = model(images, captions, match_head='itm')
            itm_score = torch.nn.functional.softmax(itm_output, dim=1)[:, 1]  # image-text matching
            itm = torch.cat([itm, itm_score.cpu()])
            sum_itm += torch.sum(itm_score)

            itc_score = torch.diagonal(model(images, captions, match_head='itc'), 0)  # image-text cosine similarity
            itc = torch.cat([itc, itc_score.cpu()])
            sum_itc += torch.sum(itc_score)

    summary = {
        'itc': sum_itc.item()/len(dataset),
        'itm': sum_itm.item()/len(dataset)
    }

    print(summary)

    data = pd.read_csv('artgraph/data/data_captions.csv', index_col=0)
    data['itm'] = itm.numpy()
    data['itc'] = itc.numpy()

    data.to_csv('data_dist.csv')

    json.dump(summary, open('artgraph/summary.json', 'w+'))
