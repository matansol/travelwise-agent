import os
import json
import csv
from dotenv import load_dotenv
from langchain.chains import SequentialChain
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import AzureChatOpenAI
from langchain.vectorstores import Qdrant

def write_token_usage_to_csv(response):
    '''gets an llm response and store the number of tokens used into a csv file'''
    token_usage = response.response_metadata['token_usage']
        # tocken_usage = {'completion_tokens': <output tokens>,
                        #  'prompt_tokens': <input tokens>,
                        #  'total_tokens': ,
                        #  'completion_tokens_details': {'accepted_prediction_tokens': 0,
                        #   'audio_tokens': 0,
                        #   'reasoning_tokens': 0,
                        #   'rejected_prediction_tokens': 0},
                        #  'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}
    input_token_counter, output_token_counter = token_usage['prompt_tokens'], token_usage['completion_tokens']
    file_path = 'token_usage.csv'
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([input_token_counter, output_token_counter])