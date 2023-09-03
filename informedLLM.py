# import os

# os.environ['KMP_DUPLICATE_LIB_OK']='True'
# from nomic.gpt4all import GPT4All

# m = GPT4All()
# m.open()
# print(m.prompt('write me a story about a lonely computer'))
# exit()
from pyChatGPT import ChatGPT
import sys

import json
import os 
from pprint import pprint
import requests
from IPython.display import HTML

result = None
result_idx = 0

def get_bing_result(query="cmu"):
    global result, result_idx
    subscription_key = os.getenv("BING_API_KEY")

    try:
        response = requests.get("https://api.bing.microsoft.com/v7.0/search", headers={'Ocp-Apim-Subscription-Key': subscription_key}, params={'q': query, 'mkt': 'en-US'})
        response.raise_for_status()
    except Exception:
        pass
    
    result = response.json()["webPages"]["value"]
    result_idx = 0
    return result[result_idx]['snippet']

def next_bing_result():
    global result, result_idx
    result_idx += 1
    return result[result_idx]['snippet']

# Initialize the model
session_token = "eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIn0..8MePDvNjp4MUgeBn.ukcVX1C91jZ7SAYJuvL7m3jp3YgmrGbqE5FAsdCh4yVcKP-1gslxC7tekucGeEw_Vq8ExjZf0DCSjk_U668rRCzpEnii1aLCep852y2YNI_sm5ZiqxtqIVDQYCGKdFD9qsqDUFrJKgnQQxTv7g9rWIqTRbL035smnvMvnCAaIVGtBb1-BvJprJhFMw72ysCGZNSOgHw1yKM10PCfTFgN4QFzoJS99QEFlmIczkdl7Mig0cPnFf-X66WFs2c4xGvvhpK_bCc_3B2w_smC8ijr9jTXc5_4RLNAq98dh78So7zaqoiUUKR0y4Vlxqu8qV8hiq98gVHclDYdZwU-REqCobXVq7fHiZuCkirW8GcDqnRlcozQOhkKs8Zo2a0YLntZWL0pVbil3l1L-Xzx2wNIr9F3IZzTazL1_Pq3DKfLF24_xnZd-cAYXKmK569f80YPlXQlJd7O-_JuXNxk4_6XDZc6hPTa3F0NpvNpkmI9U0gMtjBtXjr2euTJAe5V8GW9A84StRSyoS42KfGXyzzh87YqDZ9GLO9wSuH99NLNLjHj5hGVugBJ2QbJUVIaNr-XevR0KQjEY3I3vcnBJ7_igtE7ROLgAge0QRjQf4uvBzVii72uf2UY27c9CI5V6BU6WyDFDTDDpSnXn6SOfwfuiFEEwXRkb3hluUVc7IPIHIDehU6FcH8E72T7UjQXSM5hRZzN7pmrCANKa0WhbykrUd5xQ5S2OgNYzVnQvTOHomMWwN5kjx6238wccdoMYazRXjpyoNVB_7hEkvtG_8w_gMFRZfEygT1JMwPkmtsVvqWU6UQommbNOHKsODGT6pCKLBLlURUOs64ZjUMiKj0d-8FWPJ1OAfbOHFbq-vW2u2f9yDNshwtNBM9j6Peqay0JJBwvmizCpjRusE51ekvYA-ePODOBiuGR8LQxriXO-SeRB6B9EyN-ygfQ55sOy1_c8-g9TUG7QiRDx100kD3yJsUyhYySVcLoREkuB7fWnhFhF_PJUA5ALSVZLUTwR8oMT5PzBk_EUmK5G11Jex4-e8oVVpuyZiNduIbJ-zVrYYZFrszhPkkOK0x6_8bD1z3EIHD2VcciZ4YHeeAqv1_ZVNBaUNKseWuM2RppSSBPisoTJWXq3kovhYIfDGG3P2xVxxIEWaSQ40wVhp-yoZL8iU_rC0KlpkiLFj0zUCDC7609QfthREt9oBYAKg6ySDjPBf1PAFWoBEqDjWCQz_w-HJ8_BryienBHMu_DP2u5LJTxRUYnmHyum0Nm7Wo-qNe4LkXbeNHbFrWchvvDV9_PaZS9aD9rZW_oK40EMkpCKuJd36e3Bv9svszEGCyu3ehcqefr3LOSOrBUcDqzBBs97MtzUMQvo4ICqA5ojEWPuqd1mK8UxQnOXURdSQt9nmA_AzCmOoqxBn-9eBjhKA4xmSzZhpXzR-r5_mFSyLoZu29uCmFycAHEHlARBkRwNOWPx-35YlxHMdQuxgrnGss5EIzxtDeDkecZgpFisfrDUcwpTSxu63BMe38R_pTORlUZb9bD0_vpte85oB7ve45iF4XEQ-UScqM9SZG3Z_wv4o2NY9nG9i2HtYFZB_iaNTDwDykYff_DgBq7QTycLKO_BjU43n_FJ065C_I9sJ8JSqwYrT1sEVt_hx4RW7WGIeC1t5k9hn8oWtP_YvNSKNCbncLVn6hJksyurkgYzWEAUmrFKwNipTANoZ6esFz0S7ebakum8vdgNl5f_yj8i_kZ7PxDHmAHXTd6jo8QEzMaoqWOQ6lheVbjuA2IRV6ThwSWL_Dvg4pr5mBcxgbTscUXl0cI_ln4IFd6d_MfT6bx_RUkOVKw3ne_a0vuXaPvBAEOMHx06QorArAPMvIc2PsdxjdjmqDPppyk9m7bHx-8On7lA9kP5t3SExpA5k25TaTPiLN-iLKh-uTzvraoumKLPHvqOzy9pH9RZ61c6lIumEgB-liANh0HOy39kFSHG5mZtMzrP1glqgzHRXxB7xrTetZ5wqHmrHwvDpdWVEMPqsckBRYuktnNsTKuGmXNobJMI426d4Us5cqlK_LEK3iM5Cnt0SXVAkYgB5rBftr7qC4SwpA0gRmfWT8WjL2fvsV9aCI13Y0U9Xre_koOYJ7X6pxrZe8NzUEx5x5WR-jXgwq90riqIPGlugWTHceJF-p5mK5O8QCaCub7j5fzV_mI5G1um-tmanHWy33JF36pCEmk8LvAXSH0_Y5aIogKSmeZRrQBc25vRKfP9jnuz92mzP0pWh65wdhiB1h6yAns6WOeeGHGjoh9S4ERueq0fmDP64NNRigNZVd_Q0Vb8XnDcRr0zf8QkAx3bc0X83iwv0Fc1hgAwSRSNZ4K1TsUAd98UlZFTP_Es_FBkH-2vj8GtD1h-POAn_oKdrGhs1GRs48U66ljQ-4MvKot7A_PNRBJ_YH_JAbTFdTRDakPOSWHSKq17Lxezkt9iJtNYw8kA26GuiUJ3K4j6zQwwd_aw8dQQbK26NwlIE3QCsAohsSD1_MB5150uy2OhhxvWKj-NU-zEvBXpOc9DkRN_wCNcdbZ-aACWrVbkwEY4cDX_AZHKl7BiV_Lch58fmYYaILz6QA_OSJ0d3RQvc-gRVEt1ZrkvD0OWYsvCQ.HCQKsHjJdDcpPSfb_Loy7Q"
print("Please wait while we load ChatGPT...")
api = ChatGPT(session_token)
print("Created agent")
# question = "What is the capital of France and when was it founded?"
question = "What are the birthdays of the Nobel Prize winners in 2018?" 

question = sys.argv[1]


# reply = api.send_message(f"Your task is to ask Google queries until you have enough information to answer the question: \"{question}\" You must answer in one of two ways. If you have enough information then you preface your reply with "Answer:" and state the correct answer. Otherwise if you do not have enough information, preface your reply with "Query:" and state the query that will help you gain more information about the question. Reply with exactly one query or answer at a time.")
reply = api.send_message("Your task is to ask Google queries until you have enough information to answer the question" +
                         ": \"{your_question}\" You must answer in one of three ways. If you have enough information " + 
                         "then you preface your reply with \"Answer:\" and state the correct answer. Otherwise if you" + 
                         " do not have enough information, preface your reply with \"Query:\" and state the query tha" +
                         "t will help you gain more information about the question. If a query did not answer the que" +
                         "stion then reply with \"Next.\" If you say \"Next\" just write the word with nothing else i" + 
                         "n your reply. Reply with exactly one query or answer at a time. End your answer with a shor" + 
                         "t lesson learned from each query prefaced by \"Note:\". Provide as short, yet accurate, of an answer as possible.".format(your_question=question))

allowed_prefixes = ["Query:", "Answer:", "Next"]
allowed_attempts = 30
attempts = 0

print(f"ChatGPT: {reply['message']}")


while attempts < allowed_attempts and not reply["message"].startswith("Answer:"):
    if not reply["message"].startswith("Query:") and not reply["message"] == "Next":
        reply = api.send_message("Please preface your reply with either \"Query:\", \"Next\", or \"Answer:\"")
        continue
    attempts += 1
    our_message = ""
    if reply["message"] == "Next":
        our_message = "Google says: " + next_bing_result()
    else:
        our_message = "Google says: " + get_bing_result(question)
    print(our_message)
    print()
    reply = api.send_message(our_message)
    print(f"ChatGPT: {reply['message']}")


print(reply["message"])

