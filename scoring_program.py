import pandas as pd
import numpy as np
from thefuzz import fuzz

llama2_0 = pd.read_csv("/home/efellman/final/capstone/llama2_pipeline_answers_0.csv")
llama2_500 = pd.read_csv("/home/efellman/final/capstone/llama2_pipeline_answers_500.csv")

gpt = pd.read_csv("/home/efellman/final/capstone/chatgpt_pipeline_answers.csv")

#metrics EM, Includes EM, FuzzyWuzzy.ratio, FuzzyWuzzy.partial_ratio

llama2_metrics = dict()
llama2_metrics["EM"] = []
llama2_metrics["Includes EM"] = []
llama2_metrics["ratio"] = []
llama2_metrics["partial_ratio"] = []

gpt_metrics = dict()
gpt_metrics["EM"] = []
gpt_metrics["Includes EM"] = []
gpt_metrics["ratio"] = []
gpt_metrics["partial_ratio"] = []

for answer, guess in zip(llama2_0["answer"], llama2_0["final_answer"]):
    llama2_metrics["EM"].append(int(answer == guess))
    llama2_metrics["Includes EM"].append(int(answer in guess))
    llama2_metrics["ratio"].append(fuzz.ratio(answer, guess))
    llama2_metrics["partial_ratio"].append(fuzz.partial_ratio(answer, guess))

for answer, guess in zip(llama2_500["answer"], llama2_500["final_answer"]):
    llama2_metrics["EM"].append(int(answer == guess))
    llama2_metrics["Includes EM"].append(int(answer in guess))
    llama2_metrics["ratio"].append(fuzz.ratio(answer, guess))
    llama2_metrics["partial_ratio"].append(fuzz.partial_ratio(answer, guess))

for answer, guess in zip(gpt["answer"], gpt["final_answer"]):
    gpt_metrics["EM"].append(int(answer == guess))
    gpt_metrics["Includes EM"].append(int(answer in guess))
    gpt_metrics["ratio"].append(fuzz.ratio(answer, guess))
    gpt_metrics["partial_ratio"].append(fuzz.partial_ratio(answer, guess))

#print all in a pretty way
print("LLAMA2_0")
print("EM:", np.mean(llama2_metrics["EM"]))
print("Includes EM:", np.mean(llama2_metrics["Includes EM"]))
print("ratio:", np.mean(llama2_metrics["ratio"]))
print("partial_ratio:", np.mean(llama2_metrics["partial_ratio"]))

print("ChatGPT")
print("EM:", np.mean(gpt_metrics["EM"]))
print("Includes EM:", np.mean(gpt_metrics["Includes EM"]))
print("ratio:", np.mean(gpt_metrics["ratio"]))
print("partial_ratio:", np.mean(gpt_metrics["partial_ratio"]))