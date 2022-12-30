"""Module providing utils code for applying pretrained models from Huggingface."""
# ───────────────────────────────── Imports ────────────────────────────────── #
# Standard Library
from typing import Any, Callable, List, Literal, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import transformers as ppb  # pytorch transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

# 3rd Party

# Private

# ───────────────────────────────── Code ────────────────────────────────── #

# Setup logger
logger = logging.getLogger("evoml-explain")
logger.setLevel(logging.INFO)
console = logging.StreamHandler()
console.setLevel(level=logging.INFO)
formatter =  logging.Formatter('%(levelname)s : %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)

class PretrainedModel():
    def __init__(self, model, tokenizer, pretrained_weights: str, tag_name: str, language: str):
        self.model = model
        self.tokenizer = tokenizer
        self.pretrained_weights = pretrained_weights
        self.tag = [tag_name]
        self.language = [language]

    def get_tuple_info(self):
        return (self.model,  self.tokenizer, self.pretrained_weights)

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    def get_pretrained_weights(self):
        return self.pretrained_weights

    def get_tag(self):
        return self.tag

    def get_language(self):
        return self.language

    def update_model(self, model):
        self.model = model

    def update_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def update_pretrained_weights(self, pretrained_weights: str):
        self.pretrained_weights = pretrained_weights

    def update_tag(self, tag_name: str):
        self.tag.append(tag_name)

    def update_language(self, language: str):
        self.language.append(language)


class SupportModels():
    def __init__(self):
        self.models = []

    def add(self, model, tokenizer, pretrained_weights: str, tag_name: str, language: str):

        model_dic = {"id": self.__find_last_id(), "pretrained_model": PretrainedModel(model, tokenizer, pretrained_weights, tag_name, language)}
        self.models.append(model_dic)

    def remove(self, id: int):
        for model in self.models:
            if model["id"] == id:
                self.models.remove(model)

    def get_models(self):
        return self.models

    def __find_last_id(self):
        if not self.models:
            # empty list
            return 1
        else:
            # Find the highest id number
            return sorted(self.models, key=lambda item: item["id"])[-1]["id"] + 1


def support_pretrained_models():
    pretrained_models = SupportModels()
    # Distilbert: https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english?text=I+like+you.+I+love+you
    pretrained_models.add(ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased', "General", "English")
    pretrained_models.add(ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased', "General", "English")
    pretrained_models.add(ppb.GPT2Model, ppb.GPT2Tokenizer, 'gpt2', "General", "English")
    pretrained_models.add(ppb.AutoModelForSequenceClassification, ppb.AutoTokenizer, 'distilbert-base-uncased', "General", "English")
    pretrained_models.add(ppb.AutoModelForSequenceClassification, ppb.AutoTokenizer, 'ProsusAI/finbert', "Financial", "English")
    
    return pretrained_models.get_models()

def model_loading(pretrained_models: list, id: int):
    select_model_info = None

    for model in pretrained_models:
        if model["id"] == id:
            select_model_info = model["pretrained_model"]

    if select_model_info is None:
        return None

    model_class, tokenizer_class, pretrained_weights = select_model_info.get_tuple_info()

    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
    
    # ProsusAI/finbert
    # tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    # model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

    return (select_model_info, model, tokenizer)


def model_pipeline():
    # Dataset
    df = pd.read_csv('https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv', delimiter='\t', header=None)
    batch_1 = df[:2000]
    labels = batch_1[1]

    # Model 1:
    
    # Loading
    logger.info('Model is loading...!')
    model_info, model, tokenizer = model_loading(support_pretrained_models(), 1)
    
    # print(tokenizer('Tokenizing text is a core task in NLP'))
    # print(tokenizer.convert_ids_to_tokens(tokenizer('Tokenizing text is a core task in NLP').input_ids))
    
    # Tokenization
    logger.info('Tokenization is loading...!')
    tokenized = batch_1[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
    # def tokenize(batch):
    #     return tokenizer(batch, padding=True, truncation=True)
    
    # tokenized = batch_1[0].apply(lambda row: tokenize(row))
        
    # Padding:
    # After tokenization, tokenized is a list of sentences -- each sentences is represented as a list of tokens.
    # We want BERT to process our examples all at once (as one batch). It's just faster that way.
    # For that reason, we need to pad all lists to the same size, so we can represent the input as one 2-d array,
    # rather than a list of lists (of different lengths).
    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)

    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

    # Masking:
    # If we directly send `padded` to BERT, that would slightly confuse it.
    # We need to create another variable to tell it to ignore (mask) the padding we've added when it's processing its input.
    attention_mask = np.where(padded != 0, 1, 0)

    # Runing:
    input_ids = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask)
    
    logger.info('Pytorch is creating...!')
    
    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)

    # Only [cls] is important for us
    print(type(last_hidden_states))
    print(len(last_hidden_states))
    print(last_hidden_states)
    features = last_hidden_states[0][:, 0, :].numpy()

    # Model 2:
    # Train/Test Split
    train_features, test_features, train_labels, test_labels = train_test_split(features, 
                                                                                labels, 
                                                                                random_state=2023, 
                                                                                test_size=0.15,
                                                                                stratify=labels)
    
    logger.info('LR is loading...!')

    lr_clf = LogisticRegression()
    lr_clf.fit(train_features, train_labels)
    
    logger.info('Finished.')

    print(lr_clf.score(test_features, test_labels))
    scores = cross_val_score(lr_clf, train_features, train_labels)
    print("Logistic Regression classifier score: %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    # Logistic Regression classifier score: 0.841 (+/- 0.03)


model_pipeline()

# encoded_text = tokenizer('Tokenizing text is a core task in NLP')
# encoded_text
# tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
# tokens
# print('The vocabulary size is:', tokenizer.vocab_size)
# print('Maximum context size:', tokenizer.model_max_length)