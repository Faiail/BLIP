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

image_size = 480  # need to reshape the image for the model
seed = 1  # to have the same results

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
    #  each image needs to be transformed for the model
    transform = Compose([
        Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

    # create the dataset that needs to be iterated
    dataset = BLIPDataset(data_path='artgraph/data/data_captions.csv',
                          img_dir='artgraph/data/images-hd',
                          preprocess=transform,
                          online=False,
                          caption_column='caption')

    # the data loader is a special data structure that can iterate over a dataset with a specifi batch size
    loader = DataLoader(dataset=dataset, batch_size=8, shuffle=False)

    print('performing evaluation')

    # means
    sum_itm = 0.0
    sum_itc = 0.0

    # all the values set initially with empy tensors (lists)
    itm = torch.Tensor()
    itc = torch.Tensor()

    # for each batch of images and captions
    for images, captions in tqdm(loader):
        images = images.to(device)
        captions = captions[0]
        with torch.no_grad():
            # calculate the scores
            itm_output = model(images, captions, match_head='itm')
            itm_score = torch.nn.functional.softmax(itm_output, dim=1)[:, 1]  # image-text matching
            # append to the list and sum for the mean
            itm = torch.cat([itm, itm_score.cpu()])
            sum_itm += torch.sum(itm_score)

            itc_score = torch.diagonal(model(images, captions, match_head='itc'), 0)  # image-text cosine similarity
            # append to the list and sum for the mean
            itc = torch.cat([itc, itc_score.cpu()])
            sum_itc += torch.sum(itc_score)

    # create a map for the summary
    summary = {
        'itc': sum_itc.item()/len(dataset),
        'itm': sum_itm.item()/len(dataset)
    }

    print(summary)

    #take the whole dataset as a table
    data = pd.read_csv('artgraph/data/data_captions.csv', index_col=0)

    # insert those new columns
    data['itm'] = itm.numpy()
    data['itc'] = itc.numpy()

    # save
    data.to_csv('data_dist_hd.csv')
    json.dump(summary, open('artgraph/summary.json', 'w+'))
