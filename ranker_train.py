import argparse
argparser = argparse.ArgumentParser(description="Train a ranker!")
argparser.add_argument('--dataset', help="Location of dataset", type=str, required=True)
argparser.add_argument('--model', help="Location of model", type=str, default="bert-base-cased")
argparser.add_argument('--output', help="Location of output", type=str, default="~/models/ranker.pt")
argparser.add_argument('--batch-size', help="Batch size", type=int, default=25)
argparser.add_argument('--epochs', help="Number of epochs", type=int, default=10)
argparser.add_argument('--lr', help="Learning rate", type=float, default=0.001)
argparser.add_argument('--train-size', help="Proportion of data to use for training", type=float, default=0.8)
args = argparser.parse_args()

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import tqdm
import os
import random
import json
import torch.nn.functional as F


class QualityDataset(torch.utils.data.Dataset):
    def __init__(self, data, shrink=False):
        self.data = data
        if shrink:
            filter = (self.data['quality'].to_numpy() > 0) | (np.random.randint(2, size=len(self.data['quality'])))
            
            self.data = self.data.loc[filter == 1]
            print(f"Shrunk dataset to {len(self.data)}")
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return row['question'], row['context'], row['quality']

data = pd.read_csv(args.dataset)
split_idx = int(len(data) * args.train_size)
while data['question'][split_idx] == data['question'][split_idx - 1]:
    split_idx += 1

num_train_questions = len(set(data['question'].iloc[:split_idx]))
num_test_questions = len(set(data['question'].iloc[split_idx:]))

print(f"Train questions: {num_train_questions}\nTest questions: {num_test_questions}")

train_dataset = QualityDataset(data.iloc[:split_idx], shrink=True)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

#make test_set that pairs each question with all contexts and their qualities
test_set = dict()
for question in set(data['question'].iloc[split_idx:]):
    question_df = data.loc[data['question'] == question]
    for idx, row in question_df.iterrows():
        test_set.setdefault(row['question'],[]).append((row['context'], row['quality']))

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data.keys())
    def __getitem__(self, idx):
        question = list(self.data.keys())[idx]
        contexts, qualities = zip(*self.data[question])
        return [question] * len(contexts), contexts, qualities

test_dataset = TestDataset(test_set)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
class CustomBERTModel(nn.Module):
    def __init__(self):
          super(CustomBERTModel, self).__init__()
          self.bert = AutoModel.from_pretrained(args.model)
          self.output = nn.Linear(768, 1)

    def __call__(self, ids, mask):
        return self.forward(ids, mask)

    def forward(self, ids, mask):
        sequence_output = self.bert(
            ids, 
            attention_mask=mask)

        pooled_representation = torch.mean(sequence_output.last_hidden_state, dim=1)
        linear1_output = self.output(pooled_representation.view(-1,768)) 

        return torch.sigmoid(linear1_output)

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = CustomBERTModel()
model.to('cuda')
# criterion = nn.MSELoss()
criterion = nn.L1Loss(reduction='sum')
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
rank = 0
count = 0
for epoch in tqdm.tqdm(range(args.epochs)):
    train_loss, test_loss = 0, 0
    train_count, test_count = 0, 0
    model.train()
    for batch in tqdm.tqdm(train_dataloader): 

        questions, texts, labels = batch

        optimizer.zero_grad()   

        encoding = tokenizer.batch_encode_plus(list(zip(questions, texts)), padding=True, return_tensors="pt", max_length=512, truncation=True, add_special_tokens=True)
        input_ids = encoding['input_ids'].to('cuda')
        attention_mask = encoding['attention_mask'].to('cuda')
        outputs = model(input_ids, mask=attention_mask)
        loss = criterion(outputs, labels.float()[:,None].to('cuda') / 100)
        train_loss += loss.item()
        train_count += labels.shape[0]
        loss.backward()
        optimizer.step()
    model.eval()
    for questions, texts, quality in tqdm.tqdm(test_dataloader): 
        top_5_quality = sorted(quality, reverse=True)[:5]

        encoding = tokenizer.batch_encode_plus(list(zip(questions, texts)), padding=True, return_tensors="pt", max_length=512, truncation=True, add_special_tokens=True)
        input_ids = encoding['input_ids'].to('cuda')
        attention_mask = encoding['attention_mask'].to('cuda')
        outputs = model(input_ids, mask=attention_mask)

       

        #sort quality by outputs, items at same index should be kept together
        top_quality_found = sum([1 if x in top_5_quality else 0 for _,x in sorted(zip(outputs,quality), reverse=True, key=lambda x: x[0])][:5])
        rank += top_quality_found / 5
        print(f"Rank: {rank / len(test_set)}")
        



        loss = criterion(outputs, quality.float()[:,None].to('cuda') / 100)
        test_loss += loss.item()
        test_count += quality.shape[0]
    print(f"Train loss:\t{train_loss / train_count}\nTest loss:\t{test_loss / test_count}")
    print(f"Rank: {rank / len(test_set)}")

