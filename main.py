"""
File Name: main.py

Descripton: ...

Authors:   
    Océane Voland
    Thomas Cirillo
    Jeremy Serillon
    
Date: 11 avril 2025
Version: 1
"""


import time
import torch
import pandas as pd
import matplotlib.pyplot as plt


from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


##############################################
############# GENERAL CONSTANTS ##############
##############################################
PATH_DATA = "data/implicit-hate-corpus/"

PATH_DATA_STG1 = PATH_DATA + "/implicit_hate_v1_stg1_posts.tsv"
PATH_DATA_STG2 = PATH_DATA + "implicit_hate_v1_stg2_posts.tsv"


RANDOM_SEED = 42
NUM_LABELS = 3
MAX_LENGTH = 512 #max size of the tokenizer https://huggingface.co/GroNLP/hateBERT/commit/f56d507e4b6a64413aff29e541e1b2178ee79d67
EPOCHS = 4
BATCH_SIZE = 1

##############################################
################# FUNCTIONS ##################
##############################################


def setUpData(texts, labels,tokenizer):
    # Spliting data (80% train and 20% test)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=RANDOM_SEED
    )

    # TRAIN dataset
    train_dataset = HateSpeechDataset(
        texts=train_texts,
        labels=train_labels,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )

    # TESTING dataset
    test_dataset = HateSpeechDataset(
        texts=test_texts,
        labels=test_labels,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )


    # DATALOADER for training set
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # DATALOADER for testing set
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_dataloader, test_dataloader


def initCriterion(device):
    # Class distribution from your dataset
    class_counts = [13291, 7100, 1089]
    total = sum(class_counts)

    # Inverse frequency (optional: normalize)
    class_weights = [total / c for c in class_counts]
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Use weighted BCEWithLogitsLoss
    criterion = BCEWithLogitsLoss(weight=class_weights)

    return criterion


def train(model, train_dataloader, optimizer, device, EPOCHS=4):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    num_training_steps = EPOCHS * len(train_dataloader)
    # feel free to experiment with different num_warmup_steps
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=1, num_training_steps=num_training_steps
    )
    progress_bar = tqdm(range(num_training_steps))
    # put the model in train mode
    model.train()

    # iterate over epochs
    for epoch in range(EPOCHS):
        # iterate over batches in training set
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            # forward pass, get the outputs from the model
            outputs = model(**batch)
            # get the loss from the outputs
            loss = outputs.loss

            # do the backward pass
            loss.backward()

            # perform one step of the optimizer
            optimizer.step()

            # peform one step of the lr_scheduler, similar with the optimizer
            lr_scheduler.step()
        
            # zero the gradients, call zero_grad() on the optimizer
            optimizer.zero_grad()
            #Facilitate the update of the bar by creating artifical waiting time
            time.sleep(0.01)

            progress_bar.update(1)


def test(model, test_dataloader, device):
    # Initialize lists to store predictions and true labels
    all_predictions = []
    all_labels = []

    progress_bar = tqdm(range(len(test_dataloader)))

    # put the model in eval mode
    model.eval()
    # iterate over batches of evaluation dataset
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            # forward pass, get the outputs from the model
            outputs = model(**batch)

        #get the logits from the outputs
        logits = outputs.logits

        # use argmax to get the predicted class
        predictions = torch.argmax(logits, dim=-1)
        
        # Append predictions and true labels to the lists
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())

        progress_bar.update(1)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average="weighted", zero_division=0
    )
    # Print the results
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")


##############################################
################## CLASSES ###################
##############################################

class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return  {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
    
        
           

##############################################
################### MAIN #####################
##############################################

def main():
    # Load the data 
    data = pd.read_csv(PATH_DATA_STG1, sep = '\t')
    # Extract a subset of the data that are really going to be used
    data = data.head(20)


    id2label = {0: "not_hate", 1: "implicit_hate", 2: "explicit_hate"}
    label2id = {"not_hate": 0, "implicit_hate": 1, "explicit_hate": 2}

    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained(
        "GroNLP/hateBERT",
        num_labels=NUM_LABELS,
        id2label=id2label, 
        label2id=label2id,
    )
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("GroNLP/hateBERT")
    
    train_dataloader, test_dataloader = setUpData(data['text'], data['label'],tokenizer)

    criterion = initCriterion()

    train(model, train_dataloader, criterion, device="cuda", EPOCHS=EPOCHS)
    test(model, test_dataloader, device="cuda")

    return
    
if __name__ == "__main__":
    main()