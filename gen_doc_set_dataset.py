import argparse
argparser = argparse.ArgumentParser(description="Demo")
argparser.add_argument('--good', help="Use good supporting facts?", action='store_true')
argparser.add_argument('--hotpot', help="Location of hotpot data", type=str, required=True)
argparser.add_argument('--sentence', help="Use sentence examples?", action='store_true')
argparser.add_argument('--per-question', help="How many sets per question?", type=int, default=250)
argparser.add_argument('--min-size', help="Minimum size of set", type=int, default=1)
argparser.add_argument('--max-size', help="Maximum size of set", type=int, default=10)
argparser.add_argument('--num-questions', help="How many questions?", type=int, default=10)
argparser.add_argument('--sample-from-good', help="Proportion to sample from correct passages", type=float, default=0.5)
argparser.add_argument('--qa-model', help="QA Model", type=str, default="deepset/roberta-base-squad2")
argparser.add_argument('--output-path', help="Output location", type=str, default="~/data/doc_set_quality.csv")
args = argparser.parse_args()

"""
Currently, we just use the uniform distribution to sample size of documents
"""

import os
if not os.path.isfile(args.hotpot):
    print(f"Can't find hotpot file: {args.hotpot}")
#Used: https://huggingface.co/transformers/v2.8.0/usage.html
# to learn how to extract the answer
import tqdm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import json
import pandas as pd
import random
import torch
tokenizer = AutoTokenizer.from_pretrained(args.qa_model)
model = AutoModelForQuestionAnswering.from_pretrained(args.qa_model)

with open(args.hotpot, 'rb') as f:
    data_json = json.load(f)
    hotpot_df = pd.DataFrame.from_dict(data_json)

output_data = {"question": [], "context": [], "quality": []}

for question_idx in tqdm.tqdm(range(min(args.num_questions, len(hotpot_df)))):
    for doc_idx in range(args.per_question):
        question = hotpot_df['question'].iloc[question_idx]
        text = ""
        if doc_idx == 0:
            for c_name, c_text in hotpot_df['context'].iloc[question_idx]:
                for name, idx in hotpot_df['supporting_facts'].iloc[question_idx]:
                    if name == c_name:
                        text += " "+' '.join(c_text)
        else:
            sentences = []
            use_good = random.random() < args.sample_from_good
            good_docs = set([u for u, v in hotpot_df['supporting_facts'].iloc[question_idx]])
            for c_name, c_text in hotpot_df['context'].iloc[question_idx]:
                if not use_good or c_name in good_docs:
                    sentences += c_text
            sample_size = random.randint(args.min_size, min(len(sentences), args.max_size))
            text = ' '.join(random.sample(sentences, k=sample_size))
        inputs = tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt", max_length=514, truncation=True)
        input_ids = inputs["input_ids"].tolist()[0]
        print(question)
        print(len(input_ids))
        try:
            example_output = model(**inputs)
        except Exception as e:
            continue
        answer_start_scores, answer_end_scores = example_output.start_logits, example_output.end_logits
        output_start = torch.argmax(answer_start_scores)
        output_end = torch.argmax(answer_end_scores)
        output = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[output_start:output_end+1]))

        confidence_score = round((torch.max(torch.softmax(answer_start_scores, dim=1)) * torch.max(torch.softmax(answer_end_scores, dim=1)) * 100).item(), 2)
        correct = hotpot_df['answer'].iloc[question_idx]
        correct = tokenizer.encode_plus(correct, text, add_special_tokens=True, return_tensors="pt")

        #find correct in input_ids
        tokenized_search_space_string = tokenizer.tokenize(" " + hotpot_df['answer'].iloc[question_idx])
        tokenized_search_string = tokenizer.tokenize(hotpot_df['answer'].iloc[question_idx])
        correct_idx = 0
        correct_start = -1
        correct_end = -1
        ranges = []
        space = -1
        input = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids))
        for i, id in enumerate(tokenizer.tokenize(input)):
            if (correct_idx < len(tokenized_search_string) and tokenized_search_string[correct_idx] == id) and space != 1:
                if correct_idx == 0:
                    correct_start = i
                    space = 0
                correct_idx += 1
                if correct_idx == len(tokenized_search_string):
                    correct_end = i
                    ranges.append([correct_start, correct_end])
                    correct_idx = 0
                    space = -1
            elif (correct_idx < len(tokenized_search_space_string) and tokenized_search_space_string[correct_idx] == id) and space != 0:
                if correct_idx == 0:
                    correct_start = i
                    space = 1
                if space != 1:
                    correct_idx = 0
                    space = -1
                    continue
                correct_idx += 1
                if correct_idx == len(tokenized_search_space_string):
                    correct_end = i
                    ranges.append([correct_start, correct_end])
                    correct_idx = 0
                    space = -1
            else:
                correct_idx = 0
                space = -1
        best_s, best_e, best_v = -1, -1, -1
        for s, e in ranges:
            v = round((torch.softmax(answer_start_scores, dim=1)[0, s] * torch.softmax(answer_end_scores, dim=1)[0, e] * 100).item(), 2)
            if v > best_v:
                best_s, best_e, best_v = s, e, v

        correct_score = 0
        if len(ranges) > 0:
            correct_score = best_v
        print(f"Predicted: \t{output} {confidence_score}%\nLikelihood of guessing correctly: {correct_score}\nActual: \t{hotpot_df['answer'].iloc[question_idx]}")
        output_data["question"].append(question)
        output_data["context"].append(text)
        output_data["quality"].append(correct_score)
    output_df = pd.DataFrame.from_dict(output_data)
    output_df.to_csv(args.output_path)