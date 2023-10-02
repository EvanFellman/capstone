import argparse
argparser = argparse.ArgumentParser(description="Naive ranker tester")
argparser.add_argument('--hotpot', help="Location of hotpot data", type=str, required=True)
argparser.add_argument('--train', help="Train", type=int, default=1000)
argparser.add_argument('--test', help="Limit of hotpot data", type=int, default=1000)
argparser.add_argument('--tune', action='store_true')
argparser.add_argument('--epochs', type=int, default=1)
args = argparser.parse_args()
import random
import numpy as np
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses
from torch.utils.data import DataLoader

import os
import faiss
if not os.path.isfile(args.hotpot):
    print(f"Can't find hotpot file: {args.hotpot}")
    exit()
import torch
import tqdm
from sentence_transformers import SentenceTransformer
import json
import pandas as pd
with open(args.hotpot, 'rb') as f:
    data_json = json.load(f)
    hotpot_df = pd.DataFrame.from_dict(data_json)
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
scores_to_collect = [5, 10, 20, 50, 100]
recalls = dict()
for length in scores_to_collect:
    recalls[length] = 0


if args.tune:
    hotpot_df_train = hotpot_df.iloc[:args.train]
    train_examples = []
    print("Building training examples")

    all_good_examples = set()
    for question, context, supporting_facts in tqdm.tqdm(list(zip(hotpot_df_train['question'], hotpot_df_train['context'], hotpot_df_train['supporting_facts']))):
        for name, sentences in context:
            facts_i = set()
            for name, idx in supporting_facts:
                if name == name:
                    facts_i.add(idx)
            if len(facts_i) > 0:
                for i, sentence in enumerate(sentences):
                    #append if it's a supporting fact
                    if i in facts_i:
                        all_good_examples.add(sentence)
    for question, context, supporting_facts in tqdm.tqdm(list(zip(hotpot_df_train['question'], hotpot_df_train['context'], hotpot_df_train['supporting_facts']))):
        current_good_examples = set()
        for name, sentences in context:
            facts_i = set()
            for name, idx in supporting_facts:
                if name == name:
                    facts_i.add(idx)
            for i, sentence in enumerate(sentences):
                if i in facts_i:
                    current_good_examples.add(sentence)
                    train_examples.append(InputExample(texts=["QUESTION: " + question, "ANSWER: " + sentence], label=np.single(1) if i in facts_i else np.single(0)))
                train_examples.append(InputExample(texts=["QUESTION: " + question, "ANSWER: " + sentence], label=np.single(1) if i in facts_i else np.single(0)))
        # for sentence in all_good_examples.difference(current_good_examples):
        #     train_examples.append(InputExample(texts=["QUESTION: " + question, "ANSWER: " + sentence], label=np.single(0)))
        
        #same but sample 10 examples from other rows (context)
        # for _ in range(5):
        #     paragraphs = hotpot_df_train['context'].iloc[np.random.choice(len(hotpot_df_train['context']))]
        #     paragraph = random.choice(paragraphs)
        #     sentence = np.random.choice(paragraph[1])
        #     if sentence not in current_good_examples:
        #         train_examples.append(InputExample(texts=["QUESTION: " + question, "ANSWER: " + sentence], label=np.single(0)))
        
        #same but sample 10 examples from all_good_examples.difference(current_good_examples)
        if len(all_good_examples.difference(current_good_examples)) > 0:
            for sentence in np.random.choice(list(all_good_examples.difference(current_good_examples)), size=1000):
                train_examples.append(InputExample(texts=["QUESTION: " + question, "ANSWER: " + sentence], label=np.single(0)))
        
    train_dataset = SentencesDataset(train_examples, model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=25)
    train_loss = losses.CosineSimilarityLoss(model=model)
    print("Training")
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=args.epochs, warmup_steps=0)

hotpot_df_test = hotpot_df.iloc[max(args.train,1000):args.test+max(args.train,1000)]
index = faiss.IndexFlatIP(768)

all_encodings = np.empty((0, 768))
name_to_index = {}
name_to_index_index = 0
for context in tqdm.tqdm(hotpot_df_test['context']):
    for name, sentences in context:
        
        for idx in range(len(sentences)):
            name_to_index[f"{name}{idx}"] = name_to_index_index
            name_to_index_index += 1
        encoding = model.encode(["ANSWER: " + s for s in sentences])
        all_encodings = np.concatenate((all_encodings, encoding))
        index.add(encoding)
print("Index built")


count = 0
for question, context, supporting_facts in tqdm.tqdm(list(zip(hotpot_df_test['question'], hotpot_df_test['context'], hotpot_df_test['supporting_facts']))):
    count += 1
    q_e = model.encode("QUESTION: " + question)
    supporting = set()
    
    for name, idx in supporting_facts:
        if f"{name}{idx}" in name_to_index:
            supporting.add(name_to_index[f"{name}{idx}"])
        else:
            print(f"Can't find {name}{idx}")
    for length in scores_to_collect:
        D, I = index.search(q_e[None], length)
        
        score = len(set(I[0]).intersection(supporting))
        recalls[length] += score / len(supporting_facts)
for length in scores_to_collect:
    print(f"Recall@{length} {recalls[length] / count}")

import os
stats_file_name = "naive_ranker_stats.csv"
stats_file = None
if not os.path.isfile(stats_file_name):
    stats_file = pd.DataFrame(columns=['train', 'test', 'epochs', 'recall@5', 'recall@10', 'recall@20', 'recall@50', 'recall@100'])
else:
    stats_file = pd.read_csv(stats_file_name)
stats_file.loc[len(stats_file)] = [args.train, args.test, args.epochs if args.tune else 0, recalls[5] / count, recalls[10] / count, recalls[20] / count, recalls[50] / count, recalls[100] / count]
#save 
stats_file.to_csv(stats_file_name, index=False)

print(f"Total questions: {count}")
print(f"Total sentences: {len(name_to_index)}")