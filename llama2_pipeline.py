import os
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


## Llama 2
# Raise error if not using GPU
assert torch.cuda.is_available() == True

# initialize llama 2
llama_model_id = "/data/datasets/models/huggingface/meta-llama/Llama-2-7b-chat-hf"
llama_model = AutoModelForCausalLM.from_pretrained(llama_model_id, torch_dtype=torch.float16, device_map="auto")
llama2_tokenizer = AutoTokenizer.from_pretrained(llama_model_id)


def generate(
    message,
    tokenizer,
    max_new_tokens=1024,
    temperature=0.6,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.2,
    max_input_token_length=4096,
):
    input_ids = tokenizer.apply_chat_template([{"role": "user", "content": message}], return_tensors="pt")
    input_ids = input_ids[:, -min(input_ids.shape[1], max_input_token_length) :].to(llama_model.device)

    streamer = TextIteratorStreamer(tokenizer, timeout=None, skip_prompt=True, skip_special_tokens=True)

    t = Thread(
        target=llama_model.generate,
        kwargs={
            "input_ids": input_ids,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature,
            "num_beams": 1,
            "repetition_penalty": repetition_penalty,
        },
    )
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)

    return "".join(outputs)


def get_confidence(question, context, note):
    if note == "The note file is empty right now.":
        note = context
    else:
        note = note + " " + context

    get_confidence_prompt = f'Now you have a question-answering task. \nThe question is: {question}\nThe notes you have made so far includes: {note}\nEither answer the question or write "N/A" if you think the question is not answerable. \nAnswer:'

    is_answerable = generate(get_confidence_prompt, llama2_tokenizer)

    if (
        is_answerable.lower().find("n/a") >= 0
        or is_answerable.lower().find("apologize") >= 0
        or is_answerable.lower().find("not answerable") >= 0
    ):
        return False, note
    else:
        return is_answerable, note
    print("DIDNT GET A YES OR NO")
    return False, note


###################
# RETRIEVAL       #
###################
documents = []


def retrieve_document(query):
    global documents
    documents = retrieval_pipeline(query)
    return documents.pop(0)


def retrieve_document_next():
    if len(documents) > 1:
        return documents.pop(0)
    return []


###################


def llama2_pipeline(question):
    start_prompt = f'Write a query which will be the input to question answer retrieval machine learning pipeline to answer the question: "{question}". Please ask one query at a time and respond with the query only! \nQuery:'
    query = generate(start_prompt.format(question=question), llama2_tokenizer)
    context = retrieve_document(query)
    note = "The note file is empty right now."
    answerable, note = get_confidence(question, context, note)
    attempts_left = 5
    while answerable == False:
        context = retrieve_document_next()
        if len(context) == 0:
            if attempts_left == 0:
                answerable = False
                break
            else:
                new_query_prompt = f'NOTES: {note}\nWrite a query which will be the input to question answer retrieval machine learning pipeline to expand your knowledge to answer the question: "{question}". You already know the above notes. Please ask one query at a time and respond with the query only! \nQuery:'
                query = generate(
                    new_query_prompt.format(question=question),
                    llama2_tokenizer,
                )
                context = retrieve_document(query)
                attempts_left -= 1
        answerable, note = get_confidence(question, context, note)

    # qa_prompt = f"With all the information gathered, please answer the following question:\n{question}"
    qa_prompt = "Given the notes:" + note + f"Please answer the question: {question}"

    final_answer = answerable
    return final_answer


df = dict()
df["question"] = []
df["answer"] = []
df["final_answer"] = []

for question, answer in zip(hotpot_df_test["question"], hotpot_df_test["answer"]):
    print("Question:", question)
    print("Answer:", answer)
    print("Final answer:", llama2_pipeline(question))
    df["question"].append(question)
    df["answer"].append(answer)
    df["final_answer"].append(llama2_pipeline(question))
    # save df using pandas
    pd.DataFrame(df).to_csv("llama2_pipeline_answers.csv", index=False)
    print("=========================================")

# print("Final answer:", llama2_pipeline("What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?"))