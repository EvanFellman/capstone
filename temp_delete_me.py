import os
import faiss
import pickle
import time
import torch
import os
import numpy as np
import pandas as pd
import json
import tqdm
import argparse
from threading import Thread
from typing import Iterator

from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    QuestionAnsweringPipeline,
)

from sentence_transformers import SentenceTransformer

argparser = argparse.ArgumentParser(description="Reranker")
argparser.add_argument(
    "--hotpot",
    help="Location of hotpot data",
    type=str,
    default="/home/zjing2/capstone/finetune_roberta/hotpot_train_v1.1.json",
)
args = argparser.parse_args()
# Load model directly


# mpnet v2
retriever = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
hugging_face_model = "cross-encoder/qnli-distilroberta-base"


def get_scores(model, hotpot_df_test, question, retrieved_names):
    all_encodings = np.empty((0, 1))
    name_to_index = {}
    name_to_index_index = 0
    for context in hotpot_df_test["context"]:
        questions = []
        paragraphs = []
        for name, sentences in context:
            if name not in retrieved_names:
                continue
            name_to_index[f"{name}"] = name_to_index_index
            name_to_index_index += 1
            questions.append(question)
            paragraphs.append(" ".join(sentences))
        if len(questions) == 0:
            continue

        features = tokenizer_r(
            questions,
            paragraphs,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # to cuda
        features["input_ids"] = features["input_ids"].cuda()
        features["attention_mask"] = features["attention_mask"].cuda()
        if model != None:
            encoding = torch.nn.functional.sigmoid(model(**features).logits).detach().cpu()
            all_encodings = np.concatenate((all_encodings, encoding.detach().numpy()))
    return all_encodings, name_to_index


tokenizer_r = AutoTokenizer.from_pretrained(hugging_face_model)
model = AutoModelForSequenceClassification.from_pretrained("roberta_10000")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
model.eval()

with open(args.hotpot, "rb") as f:
    data_json = json.load(f)
    hotpot_df_test = pd.DataFrame.from_dict(data_json).iloc[-1000:]
    del data_json
print(hotpot_df_test.head(10))
count = 0
retriever_index = faiss.IndexFlatIP(768)

all_encodings = np.empty((0, 768))

# if retriever_index.pickle exists, load it
if os.path.exists("retriever_index.pickle"):
    with open("retriever_index.pickle", "rb") as f:
        retriever_index = pickle.load(f)
    with open("index_to_name_r.pickle", "rb") as f:
        index_to_name_r = pickle.load(f)
    print("Retriever Index loaded")
else:
    index_to_name_r = {}
    name_to_index_r_index = 0
    for context in tqdm.tqdm(hotpot_df_test["context"]):
        for name, sentences in context:
            index_to_name_r[name_to_index_r_index] = name
            name_to_index_r_index += 1
            encoding = retriever.encode(["ANSWER: " + " ".join(sentences)])
            all_encodings = np.concatenate((all_encodings, encoding))

    retriever_index.add(all_encodings)
    # pickle retriever_index
    with open("retriever_index.pickle", "wb") as f:
        pickle.dump(retriever_index, f, protocol=pickle.HIGHEST_PROTOCOL)
    # pickle index_to_name_r
    with open("index_to_name_r.pickle", "wb") as f:
        pickle.dump(index_to_name_r, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Retriever Index built")


def retrieval_pipeline(question):
    POSITIVE_LABEL = np.array([1])
    # retriever
    q_e = retriever.encode("QUESTION: " + question)
    _, I = retriever_index.search(q_e[None], 100)
    retrieved_set = {index_to_name_r[i] for i in I[0]}
    # reranker

    all_encodings, _ = get_scores(model, hotpot_df_test, question, retrieved_set)

    index = faiss.IndexFlatIP(1)
    index.add(all_encodings)
    length = 3
    _, I = index.search(POSITIVE_LABEL[None], length)
    retrieved_set = [index_to_name_r[i] for i in I[0]]

    # get the documents from indexes
    documents = ["" for _ in range(length)]
    for context in tqdm.tqdm(hotpot_df_test["context"]):
        for name, sentences in context:
            if name not in retrieved_set:
                continue
            idx = retrieved_set.index(name)
            documents[idx] = " ".join(sentences)
    return documents



# Finetuned RoBERTa on HotpotQA
# Raise error if not using GPU
assert torch.cuda.is_available() == True

# Initialize RoBERTa
# roberta_tokenizer = AutoTokenizer.from_pretrained("/home/zjing2/capstone/finetune_roberta/test_results/exp5/results/")
# roberta_model = AutoModelForQuestionAnswering.from_pretrained(
#     "/home/zjing2/capstone/finetune_roberta/test_results/exp5/results/"
# )


# question_answerer = QuestionAnsweringPipeline(model=roberta_model, tokenizer=roberta_tokenizer, device_map="auto")

###################
# RETRIEVAL       #
###################
documents = []


def retrieve_document(query):
    global documents
    documents = retrieval_pipeline(query)
    return documents
    return documents.pop(0)


def retrieve_document_next():
    if len(documents) > 1:
        return documents.pop(0)
    return ""


###################
# Pipeline        #
###################
def roberta_pipeline(question, threshold=0.95):
    
    score = 0.0
    context = [retrieve_document(question)]
    return context
    count = 0
    while score < threshold and count < 3:
        answer = question_answerer(question=question, context=" ".join(context))
        score = answer['score']
        context.append(retrieve_document_next())
        count += 1
    print("question:", question)
    print(f"context #doc={len(context)}:", " ".join(context))
    print("roberta answer:", answer)
    
    return answer['answer']
        
    

df = dict()
df["question"] = []
df["answer"] = []
df["final_answer"] = []

for question, answer in zip(hotpot_df_test["question"], hotpot_df_test["answer"]):
    guess = roberta_pipeline(question)
    print(guess)
    print("Question:", question)
    print("True Answer:", answer)
    print("Final answer:", guess)
    df["question"].append(question)
    df["answer"].append(answer)
    df["final_answer"].append(guess)
    # save df using pandas



    # pd.DataFrame(df).to_csv("roberta_pipeline_answers.csv", index=False)
    
    
    print("==================================================================================================")


count = 0
total_len = 0
for answer, guess in zip(df["answer"], df["final_answer"]):
    if answer in guess:
        count += 1
    total_len += 1

print(count/total_len)
