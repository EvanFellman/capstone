import argparse
argparser = argparse.ArgumentParser(description="Demo")
argparser.add_argument('--good', help="Use good supporting facts?", action='store_true')
argparser.add_argument('--example', help="Which example to use?", type=int, default=1)
argparser.add_argument('--sentence', help="Use sentence examples?", action='store_true')
args = argparser.parse_args()
#Used: https://huggingface.co/transformers/v2.8.0/usage.html
# to learn how to extract the answer

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import json
import pandas as pd
import torch
tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")




with open('hotpotqa_train.json', 'rb') as f:
    data_json = json.load(f)
    df = pd.DataFrame.from_dict(data_json)


example = args.example
example_query = df['question'].iloc[example]
text = ""
if args.good:
    for c_name, c_text in df['context'].iloc[example]:
        for name, idx in df['supporting_facts'].iloc[example]:
            if name == c_name:
                if args.sentence:
                    text += " "+c_text[idx]
                else:
                    text += " "+' '.join(c_text)
else:
    text = ' '.join(df['context'].iloc[example][0][1]) + " " + df['answer'].iloc[example]
inputs = tokenizer.encode_plus(example_query, text, add_special_tokens=True, return_tensors="pt")
input_ids = inputs["input_ids"].tolist()[0]
print(example_query)
example_output = model(**inputs)

answer_start_scores, answer_end_scores = example_output.start_logits, example_output.end_logits
output_start = torch.argmax(answer_start_scores)
output_end = torch.argmax(answer_end_scores)
output = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[output_start:output_end+1]))

confidence_score = round((torch.max(torch.softmax(answer_start_scores, dim=1)) * torch.max(torch.softmax(answer_end_scores, dim=1)) * 100).item(), 2)
correct = df['answer'].iloc[example]
correct = tokenizer.encode_plus(correct, text, add_special_tokens=True, return_tensors="pt")

#find correct in input_ids
tokenized_search_space_string = tokenizer.tokenize(" " + df['answer'].iloc[example])
tokenized_search_string = tokenizer.tokenize(df['answer'].iloc[example])
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
print(f"Predicted: \t{output} {confidence_score}%\nLikelihood of guessing correctly: {correct_score}\nActual: \t{df['answer'].iloc[example]}")