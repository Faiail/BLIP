from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json


class FastDataset(Dataset):
    """
    This class provides a data structure to manage a dataset
    """
    def __init__(self, data: pd.DataFrame, column: str = 'caption'):
        """
        @param: data: is a 'pd.DataFrame', like a table
        @param: column: is a string, indicating the column to evaluate
        """
        self.data = data
        self.column = self.data.columns.tolist().index(column)

    def __getitem__(self, item: int):
        """
        Returns the value of 'column' for the given item
        @param: item: int value that indictes which item must be taken from the dataset
        """
        return [self.data.iloc[item, self.column]]

    def __len__(self):
        """
        Returns the number of instances in the dataset
        """
        return len(self.data)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = pd.read_csv('artgraph/data/data_captions.csv', index_col=0)  # get base data

    # creates the dataset
    dataset = FastDataset(data, column='prompt')
    # create the dataloader, to iterate over batches instead of instances
    loader = DataLoader(dataset=dataset, batch_size=8, shuffle=False)

    # third party model to evaluate
    model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)
    # tokenizer to convert text into a binary vector
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token  # pad the sentence with the end-of-sentence-token

    tot_loss = 0.0
    for captions in tqdm(loader):
        captions = tokenizer(list(captions[0]), return_tensors='pt', padding=True)['input_ids'].to(device)
        loss = model(input_ids=captions, labels=captions).loss
        # compute the PERPLEXITY as e^loss
        tot_loss += torch.exp(loss).item()

    # the total perplexity is the mean of the single perplexity
    ppt = {'ppt': tot_loss/len(dataset)}
    print(ppt)
    # save
    json.dump(ppt, open('ppt_prompt.json', 'w+'))