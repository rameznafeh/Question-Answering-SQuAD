# -*- coding: utf-8 -*-
"""compute_answer.py

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1en3-aeX9lilQw4eADFZVSEJlDuBPNBmp

#Preprocessing

##Imports
"""


# The libraries we will use are imported here, in case of runtime problems
import os, shutil  #  file management
import sys 
import requests
import pandas as pd  #  dataframe management
import numpy as np  #  data manipulation

import sklearn
import copy
import glob
import re
import string
import collections

from transformers import BertTokenizerFast
from transformers import BertModel,BertPreTrainedModel
import json
import torch
import torch.nn as nn

# typing
from typing import List, Callable, Dict

"""##Dataset Download"""

import urllib.request  #  download files
import csv

project_folder = os.getcwd() #change if needed

if not os.path.exists(project_folder):
    os.makedirs(project_folder)

json_path = sys.argv[1]

with open(json_path, 'r') as json_file:
    json_data = json.load(json_file)

print("Successful extraction")

"""##Dataframe creation"""

def json_to_dataframe(input_file_path, record_path = ['data','paragraphs','qas'
                                                     ,'answers'
                                                      ],
                           verbose = 1):
    """
    input_file_path: path to the squad json file.
    record_path: path to deepest level in json file default value is
    ['data','paragraphs','qas','answers']
    verbose: 0 to suppress it default is 1
    """
    if verbose:
        print("Reading the json file")    
    file = json.loads(open(input_file_path).read())
    if verbose:
        print("processing...")
    # parsing different level's in the json file
    js = pd.io.json.json_normalize(file , record_path, [['data','title']])
    m = pd.io.json.json_normalize(file, record_path[:-1])
    r = pd.io.json.json_normalize(file, record_path[:-2])

    #combining it into single dataframe
    idx = np.repeat(r['context'].values, r.qas.str.len())
    ndx  = np.repeat(m['id'].values, m.answers.str.len())

    m['context'] = idx
    js['id'] = ndx

    main = pd.concat([m[['id','question','context']].set_index('id'),js.set_index('id')],1,sort=False).reset_index()
    if verbose:
        print("Shape of the dataframe is {}".format(main.shape))
        print("Dataframe creation done!\n")
    return main

input_file_path = json_path
record_path = ['data','paragraphs','qas','answers']
df = json_to_dataframe(input_file_path=input_file_path,record_path=record_path)

df.head()

"""##Text Cleaning"""

def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)

def white_space_fix(text):
    return ' '.join(text.split())

def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)

def lower(text):
    return text.lower()

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()

"""##Tokenizing"""

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

"""#Predict

##Load model
"""

class Bert(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        
        self.bert = BertModel(config, add_pooling_layer=False)
        # linear transformation
        self.fc = nn.Linear(config.hidden_size, 2)
        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        start_positions,
        end_positions,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs.last_hidden_state #[batch,length,dim]

        # attention_mask [batch,length]
        # token_type_ids [batch,length]

        logits = self.fc(sequence_output) #[batch,length,2]
        
        context_mask = (attention_mask-token_type_ids).unsqueeze(-1) # [batch,length,1]
        logits = logits + (context_mask + 1e-45).log()

        start_logits, end_logits = logits.split(1, dim=-1) # 2*[batch,length,1]
        start_logits = start_logits.squeeze(-1) # [batch,length]
        end_logits = end_logits.squeeze(-1) #[batch,length]

        start_loss = self.criterion(start_logits, start_positions)
        end_loss = self.criterion(end_logits, end_positions)
        loss = start_loss + end_loss

        return loss
    
    
    def get_scores(self,
        input_ids,
        attention_mask,
        token_type_ids
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )

        sequence_output = outputs.last_hidden_state

        logits = self.fc(sequence_output) #[batch,length,2]
        
        context_mask = (attention_mask-token_type_ids).unsqueeze(-1) # [batch,length,1]
        logits = logits + (context_mask + 1e-45).log()

        start_logits, end_logits = logits.split(1, dim=-1) # 2*[batch,length,1]
        start_logits = start_logits.squeeze(-1) # [batch,length]
        end_logits = end_logits.squeeze(-1) #[batch,length]

        start_score = nn.Softmax(dim=1)(start_logits)
        end_score = nn.Softmax(dim=1)(end_logits)

        return start_score, end_score

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_bert = Bert.from_pretrained(project_folder)
model_bert.to(device)
model_bert.eval()

def predict(question, context):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    with torch.no_grad():
        temp_encoding = tokenizer(context,question,truncation=True,padding=True)
        input_ids = torch.LongTensor(temp_encoding['input_ids']).unsqueeze(0).to(device)
        token_type_ids = torch.LongTensor(temp_encoding['token_type_ids']).unsqueeze(0).to(device)
        attention_mask = torch.LongTensor(temp_encoding['attention_mask']).unsqueeze(0).to(device)

        start_score, end_score = model_bert.get_scores(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        start_score = start_score.squeeze(0).cpu()
        end_score = end_score.squeeze(0).cpu()

        answer_start = torch.argmax(start_score).item()
        answer_end = torch.argmax(end_score).item()

        pred = ""
        length = start_score.size(0)

        if (answer_start == 0 or answer_end == 0 or answer_start==(length-1) or answer_end==(length-1) ) or  answer_end < answer_start or answer_end - answer_start > 20:
                pred = ""
        else:    
            input_ids.cpu()
            pred = tokenizer.decode(input_ids[0][answer_start:(answer_end+1)])

        return pred

context = df["context"]
question = df["question"]
id = df["id"]

predic = {}

for i in range(len(id)):
    con = context[i]
    ques = question[i]
    p_bert = predict(ques,con)
    predic[id[i]] = p_bert
with open("pred.json", "w") as external_file:
    json.dump(predic, external_file)

print("Prediction file ready in cwd")