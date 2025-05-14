# %% [markdown]
# # Détection de Discours Haineux Implicite avec HateBERT
# 
# ## 1. Imports et Configuration Initiale

# %%
import getpass
import os
import re
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

from tqdm.auto import tqdm
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertForSequenceClassification
from transformers import get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from IPython.display import clear_output
from torch.optim import AdamW

# %% [markdown]
# ## 2. Configuration

# %%
MODEL_NAME = 'GroNLP/hateBERT'
DATA_PATH = 'data/implicit-hate-corpus/implicit_hate_v1_stg1_posts.tsv' 
RESULTS_PATH = 'results/'
FIGURES_RESULTS_PATH = RESULTS_PATH + 'figures/'
MODELS_WEIGHTS_PATH = RESULTS_PATH + 'models_weights/'

MAX_LENGTH = 512 #max size of the tokenizer https://huggingface.co/GroNLP/hateBERT/commit/f56d507e4b6a64413aff29e541e1b2178ee79d67
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
TEST_SPLIT_SIZE = 0.2 # validation split
RANDOM_SEED = 43
NUM_LABELS = 3 # 0: not hate, 1: implicit hate, 2: explicit hate /// 

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seed for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# %% [markdown]
# ## 3. Import Data 

# %%
data = pd.read_csv(DATA_PATH, sep = '\t')
print(data)

# %% [markdown]
# ## 4. Data Set  Distribution 

# %%
class_counts = data['class'].value_counts()

# Plot
ax = class_counts.plot(kind='bar', title='Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')

for i, count in enumerate(class_counts):
    plt.text(i, count + max(class_counts)*0.01, str(count), ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(FIGURES_RESULTS_PATH + "class_distribution.png")
#plt.show()
plt.close()

# %% [markdown]
# ## 5. Data preparation (labels and text extraction and remaping)

# %%
#Can select only a subset of the data
#data = data.head(40)

# Label mappings
id2label = {0: "not_hate", 1: "implicit_hate", 2: "explicit_hate"}
label2id = {"not_hate": 0, "implicit_hate": 1, "explicit_hate": 2}


# Load data text
texts = data['post'].values

# Print raw numeric labels
print("Labels before mapping: \n", data['class'].values[:11])

# Map labels to numeric values
data['class'] = data['class'].map(label2id)
labels = data['class'].values
# Print string labels
print("Labels after mapping:  ", labels[:11])


# %% [markdown]
# # 6. Load Hate Bert model
# 
# We decide to use the Hate Bert model, a Bert model specially trained to detect hate. This model can be use from hugging face [plateforme](https://huggingface.co/transformers/v3.0.2/model_doc/auto.html).

# %%
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=id2label, 
    label2id=label2id,
    output_attentions=False,
    output_hidden_states=False
)

print(model.num_parameters())

# %% [markdown]
# # 7. Load Tokenizer
# 
# From hugging face plateforme, we can also load the tokenizer specially made for Hate Bert

# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# %% [markdown]
# # 8. Dataset Initialization

# %%
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
    
        
           

# %% [markdown]
# # 9. Dataset and DataLoader Splitting

# %% [markdown]
# To train our model, we will split the data in 3 categories as it is usually recommanded:
# - *Training*: The actual dataset that we use to train the model (weights and biases in the case of a Neural Network). The model sees and learns from this data
# - *Validation*: The sample of data used to provide an unbiased evaluation of a model fit on the training dataset while tuning model hyperparameters. 
# - *Testing*: The sample of data used to provide an unbiased evaluation of a final model fit on the training dataset.
# 
# [Source](https://medium.com/data-science/train-validation-and-test-sets-72cb40cba9e7)

# %%
# Spliting data (60% train, 20% validation and 20% test)
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=RANDOM_SEED
)
# splitting by 0.25 because: 0.25 x 0.8 = 0.2
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels, test_size=0.25, random_state=RANDOM_SEED
)

# TRAIN dataset
train_dataset = HateSpeechDataset(
    texts=train_texts,
    labels=train_labels,
    tokenizer=tokenizer,
    max_length=MAX_LENGTH
)

# VALIDATION dataset
val_dataset = HateSpeechDataset(
    texts=val_texts,
    labels=val_labels,
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

# DATALOADER for validation set
val_dataloader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# DATALOADER for testing set
test_dataloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# %% [markdown]
# # 10. Training Configuration
# 
# We use the default training configuration from the kaggle page

# %%
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# Class distribution from your dataset
class_counts = class_counts
total = sum(class_counts)

# Inverse frequency (optional: normalize)
class_weights = [total / c for c in class_counts]
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

# Use weighted BCEWithLogitsLoss
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

model.to(device)

# %% [markdown]
# # 11. Scheduler 

# %%
num_training_steps = EPOCHS * len(train_dataloader)
# feel free to experiment with different num_warmup_steps
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=1, num_training_steps=num_training_steps
)

# %% [markdown]
# # 12. Training 

# %%
def train_epoch(model, optimizer, criterion, metrics, train_dataloader, device, epoch, progress_bar):
    # Put the model in train mode
    model.train()

    # Initialize lists to store predictions and true labels
    all_predictions = []
    all_labels = []
    
    # Initialize epoch loss
    epoch_loss = 0
    epoch_metrics = dict(zip(metrics.keys(), torch.zeros(len(metrics))))

    # Use tqdm for iterating over the dataloader to see epoch progress
    train_iterator = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{EPOCHS} Training', leave=False)

    # Iterate over batches in training set
    for batch in train_iterator:
        batch = {k: v.to(device) for k, v in batch.items()}

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        target = batch["labels"]  # Get the target labels

        # Forward pass, get the outputs from the model
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=target)
        logits = outputs.logits

        # Compute the loss
        loss = criterion(logits, target)

        # Backward pass
        loss.backward()

        # Perform one step of the optimizer
        optimizer.step()

        # Learning rate scheduler step
        if 'lr_scheduler' in globals():
            lr_scheduler.step()

        # Zero the gradients
        optimizer.zero_grad()

        # Update progress bar
        progress_bar.update(1)

        # Use argmax to get the predicted class
        preds = torch.argmax(logits, dim=1)

         # Append predictions and true labels to the lists
        all_predictions.extend(preds.cpu().numpy())
        all_labels.extend(target.cpu().numpy())
        
        # Update loss
        epoch_loss += loss.item()
        
    epoch_metrics = {k: metrics[k](all_predictions, all_labels) for k in metrics.keys()}

    
    # Average the loss over all batches
    epoch_loss /= len(train_dataloader)

    # Clear the output and print epoch statistics
    clear_output()  # Clean the prints from previous epochs
    print('Train Loss: {:.4f}, '.format(epoch_loss),
          ', '.join(['{}: {:.4f}'.format(k, v) for k, v in epoch_metrics.items()]))

    return epoch_loss, epoch_metrics

# %% [markdown]
# # 13. Validation

# %%
def validation(model, criterion, metrics, val_dataloader, device, progress_bar):
    # Put the model in eval mode
    model.eval()
    
    # Initialize lists to store predictions and true labels
    all_predictions = []
    all_labels = []

    epoch_loss = 0
    epoch_metrics = dict(zip(metrics.keys(), torch.zeros(len(metrics))))

    # Use tqdm for the evaluation dataloader
    eval_iterator = tqdm(val_dataloader, desc='Evaluating Validation Set')

    # Iterate over batches of evaluation dataset
    for batch in eval_iterator:
        batch = {k: v.to(device) for k, v in batch.items()}

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        target = batch["labels"]  # Get the target labels

        with torch.no_grad():
            # Forward pass, get the outputs from the model
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=target)

            # Get the logits from the outputs
            logits = outputs.logits

        # Compute the loss
        loss = criterion(logits, target)
    
        # Use argmax to get the predicted class
        preds = torch.argmax(logits, dim=-1)
        
        # Append predictions and true labels to the lists
        all_predictions.extend(preds.cpu().numpy())
        all_labels.extend(target.cpu().numpy())
        
        # Update loss
        epoch_loss += loss.item()
    
        progress_bar.update(1)
        
    epoch_metrics = {k: metrics[k](all_predictions, all_labels) for k in metrics.keys()}
        
    # Average the loss over all batches
    epoch_loss /= len(val_dataloader)

    # Print evaluation results
    print('Eval Loss: {:.4f}, '.format(epoch_loss),
          ', '.join(['{}: {:.4f}'.format(k, v) for k, v in epoch_metrics.items()]))

    return epoch_loss, epoch_metrics

# %% [markdown]
# # 14. Plotting the training and testing

# %%
def plot_training(train_loss, val_loss, metrics_names, train_metrics_logs, test_metrics_logs, savePicture = True):
    fig, ax = plt.subplots(1, len(metrics_names) + 1, figsize=((len(metrics_names) + 1) * 5, 5))

    ax[0].plot(train_loss, c='blue', label='train')
    ax[0].plot(val_loss, c='orange', label='validation')
    ax[0].set_title('Loss')
    ax[0].set_xlabel('epoch')
    ax[0].legend()

    for i in range(len(metrics_names)):
        ax[i + 1].plot(train_metrics_logs[i], c='blue', label='train')
        ax[i + 1].plot(test_metrics_logs[i], c='orange', label='validation')
        ax[i + 1].set_title(metrics_names[i])
        ax[i + 1].set_xlabel('epoch')
        ax[i + 1].legend()

    fig.suptitle("Training result of HateBert")
    fig.savefig('results/figures/training_plot.png')
    #plt.show()
    plt.close()

def update_metrics_log(metrics_names, metrics_log, new_metrics_dict):
    '''
    - metrics_names: the keys/names of the logged metrics
    - metrics_log: existing metrics log that will be updated
    - new_metrics_dict: epoch_metrics output from train_epoch and evaluate functions
    '''
    for i in range(len(metrics_names)):
        curr_metric_name = metrics_names[i]
        metrics_log[i].append(new_metrics_dict[curr_metric_name])
    return metrics_log

# %% [markdown]
# # 15. Iterative training and validating

# %%
def training_model(model, optimizer, criterion, metrics, train_loader, val_loader, n_epochs, device):
    train_loss_log,  test_loss_log = [], []
    metrics_names = list(metrics.keys())
    train_metrics_log = [[] for i in range(len(metrics))]
    val_metrics_log = [[] for i in range(len(metrics))]

    num_training_steps = n_epochs * len(train_dataloader)

    progress_bar = tqdm(range(num_training_steps), desc="Training Progress")

    print(f"Starting training for {EPOCHS} epochs...") # Use EPOCHS from config

    for epoch in range(n_epochs):
        print("Epoch {0} of {1}".format(epoch, n_epochs))
        train_loss, train_metrics = train_epoch(model, optimizer, criterion, metrics, train_loader, device,epoch, progress_bar)

        test_loss, test_metrics = validation(model, criterion, metrics, val_loader, device, progress_bar)

        train_loss_log.append(train_loss)
        train_metrics_log = update_metrics_log(metrics_names, train_metrics_log, train_metrics)

        test_loss_log.append(test_loss)
        val_metrics_log = update_metrics_log(metrics_names, val_metrics_log, test_metrics)

        plot_training(train_loss_log, test_loss_log, metrics_names, train_metrics_log, val_metrics_log)

    progress_bar.close()
    print("Training completed.")
    return train_metrics_log, val_metrics_log


# %% [markdown]
# # 16 Evaluation metrics

# %%
def f1(preds, target):
    return f1_score(target, preds, average='macro')

def acc(preds, target):
    return accuracy_score(target, preds)

# %% [markdown]
# # 17. Main

# %%
metrics = {'ACC': acc, 'F1-weighted': f1}

model.to(device)
criterion.to(device)

train_metrics_log, test_metrics_log = training_model(model, optimizer, criterion, metrics, train_dataloader, val_dataloader, n_epochs=EPOCHS, device=device)

# save model weights
if not os.path.exists(MODELS_WEIGHTS_PATH):
    os.mkdir(MODELS_WEIGHTS_PATH)
torch.save(model.state_dict(), MODELS_WEIGHTS_PATH + 'base_model.pth')

# %% [markdown]
# # 18. Testing 

# %%
def testing(model, metrics, test_dataloader, device, progress_bar):
    # Put the model in eval mode
    model.eval()
    
    # Initialize lists to store predictions and true labels
    all_predictions = []
    all_labels = []

    epoch_metrics = dict(zip(metrics.keys(), torch.zeros(len(metrics))))

    # Use tqdm for the evaluation dataloader
    eval_iterator = tqdm(test_dataloader, desc='Evaluating Test Set')

    # Iterate over batches of evaluation dataset
    for batch in eval_iterator:
        batch = {k: v.to(device) for k, v in batch.items()}

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        target = batch["labels"]  # Get the target labels

        with torch.no_grad():
            # Forward pass, get the outputs from the model
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=target)

            # Get the logits from the outputs
            logits = outputs.logits
    
        # Use argmax to get the predicted class
        preds = torch.argmax(logits, dim=-1)
        
        # Append predictions and true labels to the lists
        all_predictions.extend(preds.cpu().numpy())
        all_labels.extend(target.cpu().numpy())

        progress_bar.update(1)

    # Compute metrics on the entire dataset
    epoch_metrics = {k: metrics[k](all_predictions, all_labels) for k in metrics.keys()}
        
    return epoch_metrics

# %%
def testing_process(model, metrics, test_dataloader, device):
    # Use tqdm for the evaluation dataloader
    num_testing_steps = len(test_dataloader)
    progress_bar = tqdm(range(num_testing_steps), desc="Testing Progress")

    test_metrics = testing(model, metrics, test_dataloader, device, progress_bar)
    
    progress_bar.close()
    print("Training completed.")
    return test_metrics

# %%
def saveMetrics(metrics, title):
    with open(RESULTS_PATH + "testing_results.txt", "w") as f:
        f.write(f"{title} \n")
        for name, score in metrics.items():
            f.write(f"- {name}, : {score} \n")

# %%
def showMetrics():
    with open(RESULTS_PATH + "testing_results.txt") as f:
        print(f.read())

# %%
test_metrics = testing_process(model, metrics, test_dataloader, device)

# %% [markdown]
# Save the testing results in a file

# %%
saveMetrics(test_metrics, "Testing results metrices")
showMetrics()

# %% [markdown]
# # 19. Inference

# %%
def saveInference(string):
    with open(RESULTS_PATH + "inference_results.txt", "w") as f:
      f.write(string)

# %%
def classification(example_text, example_label, show = False, save= True): 
    #Using the HateBertDataLoader is a bit overkilled, we can just tokenize the input
    encoded_input = tokenizer(
        example_text,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to(device)
    
    model.eval()
    
    with torch.no_grad():
        # Forward pass, get the outputs from the model
        outputs = model(**encoded_input)
        
        # Get the logits from the outputs
        logits = outputs.logits
    
    # Use argmax to get the predicted class
    preds = torch.argmax(logits, dim=-1)

    res_str =  f"Example sentence: {example_text}\n ---- Model classification: {id2label[int(preds)]}\n ---- Real classification: {id2label[example_label]}\n"
    if show: print(res_str)
    if save: saveInference(res_str)

# %%
#Write here an example sentence
example_text = "I like train"
#Determine the type of hate of your sentence 
example_label = "not_hate" 
classification(example_text, label2id[example_label], True)

# %%
#Write here an example sentence
example_text = "White people should all die"
#Determine the type of hate of your sentence 
example_label = "implicit_hate" 
classification(example_text, label2id[example_label], True)


