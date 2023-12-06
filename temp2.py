
# print("Final answer:", llama2_pipeline("What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?"))import os
import faiss
import pickle
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
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    AutoModelForSequenceClassification,
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


def get_scores(model, hotpot_df_test, question, retrieved_texts, retrieved_names):
    all_encodings = np.empty((0, 1))
    index_to_name = {}
    questions = []
    paragraphs = []
    for doc, name in zip(retrieved_texts, retrieved_names):
        questions.append(question)
        paragraphs.append(" ".join(doc))
        index_to_name[len(index_to_name)] = name
    
    

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
    return all_encodings, index_to_name


tokenizer_r = AutoTokenizer.from_pretrained(hugging_face_model)
model = AutoModelForSequenceClassification.from_pretrained("roberta_10000")
# model = AutoModelForSequenceClassification.from_pretrained(hugging_face_model)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
model.eval()

with open(args.hotpot, "rb") as f:
    data_json = json.load(f)
    hotpot_df_test = pd.DataFrame.from_dict(data_json).iloc[1000:2000]
    del data_json

count = 0
retriever_index = faiss.IndexFlatIP(768)

all_encodings = np.empty((0, 768))

# if retriever_index.pickle exists, load it
if os.path.exists("all_encodings_v2.npy"):
    with open("retriever_index_v2.pickle", "rb") as f:
        retriever_index = pickle.load(f)
    with open("index_to_text_retriever.pickle", "rb") as f:
        index_to_text_r = pickle.load(f)
    with open("index_to_name_retriever.pickle", "rb") as f:
        index_to_name_r = pickle.load(f)
    print("Retriever Index loaded")
    #load all_encodings
    all_encodings = np.load("all_encodings_v2.npy")
else:
    index_to_text_r = {}
    index_to_name_r = {}
    name_to_index_r = {}
    name_to_index_r_index = 0
    for context in tqdm.tqdm(hotpot_df_test["context"]):
        for name, sentences in context:
            index_to_text_r[name_to_index_r_index] = " ".join(sentences)
            index_to_name_r[name_to_index_r_index] = name
            name_to_index_r_index += 1
            encoding = retriever.encode(["ANSWER: " + " ".join(sentences)])
            all_encodings = np.concatenate((all_encodings, encoding))
    #save all_encodings to file
    np.save("all_encodings_v2.npy", all_encodings)

    # print(all_encodings.shape)
    retriever_index.add(all_encodings)
    # pickle retriever_index
    with open("retriever_index_v2.pickle", "wb") as f:
        pickle.dump(retriever_index, f, protocol=pickle.HIGHEST_PROTOCOL)
    # pickle index_to_name_r
    with open("index_to_text_retriever.pickle", "wb") as f:
        pickle.dump(index_to_text_r, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open("index_to_name_retriever.pickle", "wb") as f:
        pickle.dump(index_to_name_r, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open("name_to_index_retriever.pickle", "wb") as f:
        pickle.dump(name_to_index_r, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Retriever Index built")


def retrieval_pipeline(question, supporting_facts=None):
    POSITIVE_LABEL = np.array([1])
    # retriever
    q_e = retriever.encode("QUESTION: " + question)
    # print(q_e)

    #dot product q_e with each row of all_encodings
    dot_prod = np.dot(all_encodings, q_e)
    I = dot_prod.argsort()[-100:][::-1]
    # _, I = retriever_index.search(q_e[None], 100)
    retrieved_set = {index_to_text_r[i] for i in I}
    retrieved_names = {index_to_name_r[i] for i in I}
    print("retrieved set:", retrieved_names)
    if supporting_facts != None:
        
        retrieved_names = [index_to_name_r[i] for i in I if index_to_name_r[i] in supporting_facts]
        return len(retrieved_names)
    # reranker

    # retrieved_set = list(index_to_name_r.values())
    # if supporting_facts != None:
        
    #     retrieved_set = [index_to_name_r[i] for i in I[0] if index_to_name_r[i] in supporting_facts]
    #     return len(retrieved_set)

    all_encodings_reranker, asdf = get_scores(model, hotpot_df_test, question, retrieved_set, retrieved_names)
    # all_encodings_reranker = all_encodings_reranker.flatten()
    index = faiss.IndexFlatL2(1)
    index.add(all_encodings_reranker)
    length = 3
    _, I = index.search(POSITIVE_LABEL[None], length)
    # I = [all_encodings_reranker.argsort()[-length:][::-1]]
    retrieved_set = [asdf[i] for i in I[0]]
    # get the documents from indexes
    # documents = ["" for _ in range(length)]
    # for context in tqdm.tqdm(hotpot_df_test["context"]):
    #     for name, sentences in context:
    #         if name not in retrieved_set:
    #             continue
    #         idx = retrieved_set.index(name)
    #         documents[idx] = " ".join(sentences)
    return retrieved_set


## Llama 2
# Raise error if not using GPU
assert torch.cuda.is_available() == True

# initialize llama 2
# llama_model_id = "/data/datasets/models/huggingface/meta-llama/Llama-2-7b-chat-hf"
# llama_model = AutoModelForCausalLM.from_pretrained(llama_model_id, torch_dtype=torch.float16, device_map="auto")
# llama2_tokenizer = AutoTokenizer.from_pretrained(llama_model_id)



###################
# RETRIEVAL       #
###################
documents = []


def retrieve_document(query):
    # global documents
    documents = retrieval_pipeline(query)
    return documents


df = dict()
df["question"] = []
df["answer"] = []
df["final_answer"] = []

# recall = 0
# for i in tqdm.tqdm(range(100)):
#     hotpot_df_test_r = hotpot_df_test.iloc[i]
#     question = hotpot_df_test_r["question"]
#     answer = hotpot_df_test_r["answer"]
#     supporting_facts = hotpot_df_test_r["supporting_facts"]
#     #get the firsts

#     supporting_facts = [x[0] for x in supporting_facts]
#     # print(question)
#     print("supporting facts:", supporting_facts)

#     n = retrieval_pipeline(question, supporting_facts=supporting_facts) 
#     recall += n / len(set(supporting_facts))
# recall /= 100
# print(f"Recall: {recall}")
recalls = {3: 0}
for question, context, supporting_facts in tqdm.tqdm(list(zip(hotpot_df_test['question'], hotpot_df_test['context'], hotpot_df_test['supporting_facts']))):
    #retriever
    q_e = retriever.encode("QUESTION: " + question)
    D, I = retriever_index.search(q_e[None], 100) #change to 100

    retrieved_set = {index_to_name_r[i] for i in I[0]}
    supporting = set()
    supporting_len = len(set([name for name, _ in supporting_facts]))

    
    for name, idx in supporting_facts:
        if f"{name}" in name_to_index_r:
            supporting.add(name_to_index_r[f"{name}"])
        else:
            print(f"Can't find {name}{idx}")
    score = len(set(I[0]).intersection(supporting))
    #reranker
    count += 1

    if args.no_reranker:
        continue
    all_encodings, name_to_index = get_scores(model, hotpot_df_test, question, retrieved_set)
    supporting = set()
    for name, idx in supporting_facts:
        if f"{name}" in name_to_index:
            supporting.add(name_to_index[f"{name}"])
        else:
            print(f"Can't find {name}{idx}")
    index = faiss.IndexFlatL2(1)
    index.add(all_encodings)
    for length in [3]:
        D, I = index.search(np.array([1])[None], length)
        
        score = len(set(I[0]).intersection(supporting))
        recalls[length] += 0 if supporting_len == 0 else score / supporting_len
        # recalls[length] += 0 if len(supporting) == 0 else score / len(supporting)
# print(retrieve_document("Zhi Jing studies where?"))
# print(retrieve_document('Alan Douglas Ruck, is an American actor, he played Cameron Frye, Bueller\'s hypochondriac best friend in John Hughes\' "Ferris Bueller\'s Day Off", released in which year?'))

# for question, answer in zip(hotpot_df_test["question"][::-1], hotpot_df_test["answer"][::-1]):
#     print(question)
#     print(answer)
#     print()
#     context = retrieve_document(question)
#     print(context)



    # pd.DataFrame(df).to_csv("llama2_pipeline_answers_backwards.csv", index=False)
    
    
    # print("==================================================================================================")

# print("Final answer:", llama2_pipeline("What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?"))