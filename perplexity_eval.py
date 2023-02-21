from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json


class FastDataset(Dataset):
    def __init__(self, data: pd.DataFrame, column: str = 'caption'):
        self.data = data
        self.column = self.data.columns.tolist().index(column)

    def __getitem__(self, item):
        return [self.data.iloc[item, self.column]]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = pd.read_csv('artgraph/data/data_captions.csv', index_col=0)
    dataset = FastDataset(data, column='prompt')
    loader = DataLoader(dataset=dataset, batch_size=8, shuffle=False)

    model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    tot_loss = 0.0
    for captions in tqdm(loader):
        captions = tokenizer(list(captions[0]), return_tensors='pt', padding=True)['input_ids'].to(device)
        loss = model(input_ids=captions, labels=captions).loss
        tot_loss += torch.exp(loss).item()

    ppt = {'ppt': tot_loss/len(dataset)}
    print(ppt)
    json.dump(ppt, open('ppt_prompt.json', 'w+'))