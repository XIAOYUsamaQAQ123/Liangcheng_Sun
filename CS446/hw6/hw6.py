import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    GPT2ForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from typing import Tuple, Dict
from tqdm import tqdm
from datasets import load_dataset, Dataset
import math
import random


from hw6_utils import prompt_zero, load_classification_data, eval_model






## Problem: Simple Q-Learning

### (a)
def calculate_q_update(q_value, reward, next_max_q, alpha, gamma):
    """
    Calculate the updated Q-value using the Q-learning formula.
    
    This function implements the core Q-learning update equation:
    Q(s,a) ← Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
    
    Parameters:
        q_value (float): Current Q-value for state s and action a
        reward (float): Reward received after taking action a in state s
        next_max_q (float): Maximum Q-value for the next state s' across all actions
        alpha (float): Learning rate (between 0 and 1)
        gamma (float): Discount factor (between 0 and 1)
    
    Returns:
        float: Updated Q-value for state s and action a
    

    """
    # TODO: Calculate the temporal difference (TD) error
    # TD error = reward + (discount factor * max next Q-value) - current Q-value
   
    return q_value + alpha * (reward+gamma * next_max_q-q_value)
    # TODO: Update the Q-value by adding a fraction (determined by alpha) of the TD error
    
    
    



# Problem (b)

def select_best_action(q_values):
    """
    Select the action with the highest Q-value (greedy policy).
    
    Parameters:
        q_values (list): List of Q-values for all possible actions in the current state
                         For example, [0.5, 0.2] where 0.5 is Q-value for action 0 (right)
                         and 0.2 is Q-value for action 1 (down)
    
    Returns:
        int: Index of the action with the highest Q-value
             (0 for right, 1 for down in our 2x2 grid world)
    
    """
    # TODO: Implement the greedy policy
    return int(np.argmax(q_values))
    
    




## Problem (c)
def simple_training_loop(episodes, alpha, gamma):
    """
    Implement a simple Q-learning training loop for the 2x2 grid world.
    
    This function trains an agent using Q-learning in the simple 2x2 grid world
    where the agent can move right or down. The goal is to reach the bottom-right
    corner (state 3).
    
    Parameters:
        episodes (int): Number of episodes to train the agent
        alpha (float): Learning rate (between 0 and 1)
        gamma (float): Discount factor (between 0 and 1)
    
    Returns:
        numpy.ndarray: The final Q-table after training
    
    Example:
        >>> simple_training_loop(10, 0.1, 0.9)
        # Returns the trained Q-table after 10 episodes
    """
    # Initialize Q-table with zeros for 4 states and 2 actions
    q_table = np.zeros((4, 2))
    
    # TODO: complete the training loop
    for episode in range(episodes):
        state = 0  # Start at initial state (top-left)
        done = False
        
        while not done:
            # TODO: Choose action with highest Q-value (greedy policy)
            action = select_best_action(q_table[state])
            
            # TODO: Get next state (right = go to the state on the right, down = go to the state below)
            x, y = state % 2, state // 2
            if action == 0:
                x = min(x + 1, 1)
            else:
                y = min(y + 1, 1)
            next_state = y * 2 + x
            
            # TODO: Get reward, -1 for each step, 10 for reaching the goal (state 3)
            if next_state == 3:
                reward = 10
                done = True
            else:
                reward = -1
           
            # TODO: Update Q-table using the Q-learning formula
            curr_q = q_table[state, action]
            next_max = np.max(q_table[next_state])
            q_table[state, action] = calculate_q_update(curr_q, reward, next_max, alpha, gamma)
            
            # TODO: Move to next state
            state = next_state
    
    
    return q_table



## Problem: GPT-2 Finetuning
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED);  random.seed(SEED)
torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
# --------------------------------------------------------------------
# Do Not modify the following Constants
# --------------------------------------------------------------------
MODEL_NAME = "gpt2"
TOKENIZER = GPT2Tokenizer.from_pretrained(MODEL_NAME)
TOKENIZER.pad_token = TOKENIZER.eos_token
TOKENIZER.pad_token_id = TOKENIZER.eos_token_id
YES_ID, NO_ID = TOKENIZER.encode(" yes")[0], TOKENIZER.encode(" no")[0]

MAX_LEN = 200   # maximum total length (prompt + answer + pads)



### Problem (a) 
def build_prompt_with_answer(ex):
    """
    Create a training prompt string paired with the correct yes/no answer.

    This function takes a single example `ex` from the dataset, where:
      - `ex["content"]` is the review text (a string).
      - `ex["label"]` is 1 for positive or 0 for negative sentiment.

    It should return a dict with key `"text"` whose should use the same
    prompt as in evaluation (`prompt_zero`) followed directly by the
    correct answer token.  This can be used as training data to finetune gpt2
    in seq2seq setting, so that eval_seq2seq lead to better accuracy.

    Parameters
    ----------
    ex : dict
        A dataset record containing:
          • "content": the review text.
          • "label": integer 1 (positive) or 0 (negative).

    Returns
    -------
    dict
        {"text": "<prompt_with_answer>"} 
    """
    answer = "yes" if ex["label"] == 1 else "no"
    return {"text": prompt_zero(ex["content"]) + answer}



### Problem (b)

def tokenize_seq2seq(batch):
    """
    Tokenize examples for seq2seq fine‑tuning of GPT‑2.

    Parameters
    ----------
    batch : dict[str, list]
        A batch from a dataset with keys `"content"` (list of review texts)
        and `"label"` (list of 0/1 sentiment labels).

    Returns
    -------
    dict[str, list]
        A mapping suitable for `Trainer`, containing:
          - "input_ids": List[List[int]]
          - "attention_mask": List[List[int]]
          - "labels": List[List[int]]
    """
    texts = []
    for txt, lab in zip(batch["content"], batch["label"]):
        texts.append(build_prompt_with_answer({"content": txt, "label": lab})["text"])

    tokenized = TOKENIZER(
        texts,
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length",
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized
    
def train_seq2seq(train, lm):
    """
    Fine‑tune GPT‑2 in a seq2seq setup for sentiment classification.

    Parameters
    ----------
    train : Dataset
        A Hugging Face Dataset with columns "content" and "label".
    lm : GPT2LMHeadModel
        The pretrained GPT‑2 model (with language model head) to be fine‑tuned.

    Returns
    -------
    """

    ds2 = train.map(
        tokenize_seq2seq,
        batched=True,
        remove_columns=["content","label"]
    )
    data_collator = DataCollatorWithPadding(tokenizer=TOKENIZER)
    args = TrainingArguments(
        output_dir="./_seq2seq",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        learning_rate=2e-4,
        save_strategy="no",
        logging_strategy="no",
        report_to=[],
    )
    trainer = Trainer(
        model=lm,
        args=args,
        train_dataset=ds2,
        data_collator=data_collator,
        tokenizer=TOKENIZER,
    )
    trainer.train()
    return trainer



### Problem (c)

def tokenize_clf(batch):
    """
    Tokenize a batch of examples for GPT‑2 classification fine‑tuning.

    Parameters
    ----------
    batch : dict[str, list]
        A batch from the dataset containing:
          - "content": list of review strings
          - "label": list of integer labels (0 or 1)
    
    Returns
    -------
    dict[str, list]
        A mapping with keys:
          - "input_ids": token IDs
          - "attention_mask": attention masks
          - "labels": integer labels for loss computation
    """
    toks = TOKENIZER(
        batch["content"],
        truncation=True,
        padding=False,
        max_length=MAX_LEN,
    )
    toks["labels"] = batch["label"]
    return toks


def train_clf(train: Dataset, model: GPT2ForSequenceClassification) :
    """
    Fine‑tune the provided GPT‑2 classification head on the sentiment dataset.

    Parameters
    ----------
    train : Dataset
        A Dataset with fields "content" (text) and "label" (0/1).
    model : GPT2ForSequenceClassification
        A GPT‑2 model with a classification head, already on DEVICE.

    Returns
    -------
    """
    ds_tok = train.map(tokenize_clf, batched=True, remove_columns=["content", "label"])
    data_collator = DataCollatorWithPadding(tokenizer=TOKENIZER)
    args = TrainingArguments(
        output_dir="./_clf",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        learning_rate=2e-4,
        save_strategy="no",
        logging_strategy="no",
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tok,
        data_collator=data_collator,
        tokenizer=TOKENIZER,
    )
    trainer.train()
    return trainer

###### You may want to use the following code to debug your implementation
# # test the build prompt function
# print(build_prompt_with_answer({"content": "I love this movie!", "label": 1})["text"])

# # initialize the dataset
# train_set = load_classification_data("classification_train.txt")

# # initialize the seq2seq model
# lm_seq2seq = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
# # train seq2seq
# train_seq2seq(train_set, lm_seq2seq)
# # evaluate seq2seq on the training set to make sure it works
# seq_loss, seq_acc = eval_model(lm_seq2seq)


# # initialize the classification model
# lm_clf = GPT2ForSequenceClassification.from_pretrained(
#     MODEL_NAME, num_labels=2, pad_token_id=TOKENIZER.pad_token_id).to(DEVICE)
# # train classification model
# train_clf(train_set, lm_clf)
# # evaluate classification model on the training set to make sure it works
# clf_loss, clf_acc = eval_model(lm_clf, tokenize_clf)









## Problem: Transformer Attnetion


def scaled_dot_product_attention(q, k, v):
    '''
    Compute the attention weights and output.
    Arguments:
        q: query tensor of shape (batch_size, seq_len, d_k)
        k: key tensor of shape (batch_size, seq_len, d_k)
        v: value tensor of shape (batch_size, seq_len, d_k)
    Returns:
        output: attention output tensor of shape (batch_size, seq_len, d_k)
    '''
    scores = torch.matmul(q, k.transpose(-2, -1))/ math.sqrt(q.size(-1))
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, v)
     
 
 
 
 
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        '''
        Initializes the layers of your multi-head attention module by calling the superclass
        constructor and setting up the layers.
        Arguments:
            d_model: the number of features in the input
            num_heads: the number of attention heads
        The layers of your multi-head attention module (in order) should be
        - A linear layer (torch.nn.Linear) for the query
        - A linear layer (torch.nn.Linear) for the key
        - A linear layer (torch.nn.Linear) for the value
        - A linear layer (torch.nn.Linear) for the output
        '''
         
         
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
 
    def forward(self, q, k, v):
        '''
        A forward pass of your multi-head attention module.
        Arguments:
            q: query tensor of shape (batch_size, seq_len, d_model)
            k: key tensor of shape (batch_size, seq_len, d_model)
            v: value tensor of shape (batch_size, seq_len, d_model)
        Returns:
            output: attention output tensor of shape (batch_size, seq_len, d_model)
         '''
        b, l, _ = q.size()
        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)
        Q = Q.view(b, l, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(b, l, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(b, l, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        weights = F.softmax(scores, dim=-1)
        head_out = torch.matmul(weights, V)
        concat = head_out.transpose(1, 2).contiguous().view(b, l, self.d_model)
        return self.w_o(concat)
    