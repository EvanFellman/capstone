
import argparse
argparser = argparse.ArgumentParser(description="Reranker")
argparser.add_argument('--hotpot', help="Location of hotpot data", type=str, default="/home/efellman/data/hotpot_dev_distractor_v1.json")
argparser.add_argument('--train', help="Train", type=int, default=0)
argparser.add_argument('--test', help="Limit of hotpot data", type=int, default=1000)
argparser.add_argument('--load-reranker', type=str, default="roberta_10000")
args = argparser.parse_args()
# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample
import faiss

import torch
import numpy as np
import pandas as pd
import json
import tqdm
#mpnet v2
retriever = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
hugging_face_model = "cross-encoder/qnli-distilroberta-base"

def get_scores(model, hotpot_df_test, question, retrieved_names):
    all_encodings = np.empty((0,1))
    name_to_index = {}
    index_to_text = {}
    name_to_index_index = 0
    for context in tqdm.tqdm(hotpot_df_test['context']):
        questions = []
        paragraphs = []
        for name, sentences in context:
            if name not in retrieved_names:
                continue
            name_to_index[name] = name_to_index_index
            index_to_text[name_to_index_index] = " ".join(sentences)
            name_to_index_index += 1
            questions.append(question)
            paragraphs.append(" ".join(sentences))
        if len(questions) == 0:
            continue
        
        features = tokenizer(questions, paragraphs, padding="max_length", truncation=True, return_tensors="pt")
        #to cuda
        features['input_ids'] = features['input_ids'].cuda()
        features['attention_mask'] = features['attention_mask'].cuda()
        if model != None:
            encoding = torch.nn.functional.sigmoid(model(**features).logits).detach().cpu()
            if "deberta" in hugging_face_model:
                encoding = torch.softmax(encoding, -1)[:,0][:,None]
            all_encodings = np.concatenate((all_encodings, encoding.detach().numpy()))
    return all_encodings, name_to_index, index_to_text

 
tokenizer = AutoTokenizer.from_pretrained(hugging_face_model)
model = AutoModelForSequenceClassification.from_pretrained("roberta_10000")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

with open(args.hotpot, 'rb') as f:
    data_json = json.load(f)
    hotpot_df_test = pd.DataFrame.from_dict(data_json).iloc[:1000]

count = 0
retriever_index = faiss.IndexFlatIP(768)

all_encodings = np.empty((0, 768))

index_to_name_r = {}
name_to_index_r = {}
name_to_index_r_index = 0
for context in tqdm.tqdm(hotpot_df_test['context']):
    #context is a list of 10 paragraphs and names
    for name, sentences in context:
        index_to_name_r[name_to_index_r_index] = name
        name_to_index_r[name] = name_to_index_r_index
        name_to_index_r_index += 1
        encoding = retriever.encode(["ANSWER: " + " ".join(sentences)])
        all_encodings = np.concatenate((all_encodings, encoding))
retriever_index.add(all_encodings)
print("Retriever Index built")

def retrieval_pipeline(question):
    POSITIVE_LABEL = np.array([1])
    q_e = retriever.encode("QUESTION: " + question)
    _, I = retriever_index.search(q_e[None], 100) 

    retrieved_set = {index_to_name_r[i] for i in I[0]}

    all_encodings, _, index_to_text = get_scores(model, hotpot_df_test, question, retrieved_set)
    
    index = faiss.IndexFlatL2(1)
    index.add(all_encodings)

    _, I = index.search(POSITIVE_LABEL[None], 3)
    output = []
    for i in I[0]:
        output.append(index_to_text[i])
    return output

row = hotpot_df_test.iloc[0]
question = row["question"]
print("Question:", question)
retrieval_pipeline(question)