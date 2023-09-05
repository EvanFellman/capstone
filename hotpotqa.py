import pandas as pd
import json
import openai
import requests
import os
from transformers import pipeline
import argparse

argparser = argparse.ArgumentParser(description="Ask a question and you shall receive an answer.")
argparser.add_argument('--query', type=str, default="Where is Evan Fellman studying?", help="The question to ask")
argparser.add_argument('--local', action='store_true', help="Use offline OPT IML 7B instead of ChatGPT")
argparser.add_argument('--threshold', type=float, default=1e-3, help="The threshold us to consider the QA model's response as a confident answer")
args = argparser.parse_args()

with open('hotpotqa_train.json', 'rb') as f:
    data_json = json.load(f)
df = pd.DataFrame.from_dict(data_json)
pipe = pipeline("question-answering", model="deepset/roberta-base-squad2")
openai.api_key = os.getenv("OPENAI_API_KEY")
chatgpt = not args.local

example_query = df['question'][0]

# Initialize the model
generator = None
if chatgpt:
    def generatorFunc (prompt, max_length, temperature, do_sample, top_k, top_p, repetition_penalty, num_return_sequences):  
        return [{'generated_text': openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ]
            )['choices'][0]['message']['content']}]
    generator = generatorFunc
else:
    generator = pipeline('text-generation', model="facebook/opt-iml-max-1.3b")


notes = []
ready = False
article_iter = iter(df['context'][0])

while not ready:
    prompt = ""
    if len(notes) == 0:
        with open("start_prompt.txt", "r") as f:
            prompt = f.read().format(question=example_query)
    else:
        with open("get_query_with_notes_prompt.txt", "r") as f:
            prompt = f.read().format(question=example_query, notes="\n".join(notes))
    response = generator(prompt, max_length = 200, temperature=0.75, do_sample=True, top_k=100, top_p=0.95, repetition_penalty=1, num_return_sequences=1)[0]['generated_text']
    query = response.split("\n")[-1].split("Query:")[-1].strip()

    print(f"LLM Query: Get first paragraph!")
    title, paragraph = next(article_iter, ["END OF THE STORY", "STOP"])

    print(f"  Answer: " + title + ": " + "".join(paragraph))
    content = False
    query_answer = pipe({'question': query, 'context': title + ": " + "".join(paragraph)})
    while not content: 
        #using pretrained QA model to validate Bing results
        if query_answer['score'] >= args.threshold:
            print("  Content!")
            content = True
            break

        print("  Not content yet...")
        title, paragraph = next(article_iter, ["END OF THE STORY", "STOP"])
        if title == "END OF THE STORY" and paragraph == "STOP":
            print("  NO MORE STORY!")
            break

        query_answer = pipe({'question': query, 'context': title + ": " + "".join(paragraph)})
        print(f"  Response: " + title + ": " + "".join(paragraph))
    
    notes.append(title + ": " + "".join(paragraph))
    answer_to_question = pipe({'question': example_query, 'context': "\n".join(notes)})
    if answer_to_question['score'] >= args.threshold:
        print(f'Answer: {answer_to_question["answer"]}')
        exit()

print("True Answer:", df['answer'][0])
