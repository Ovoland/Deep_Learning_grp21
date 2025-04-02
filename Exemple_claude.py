# Import necessary libraries
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load and prepare the dataset
class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
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
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 2. Load and initialize the tokenizer
def initialize_model_and_tokenizer(num_labels=3):  # Adjust num_labels based on your classification needs
    # Load the HateBERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained("GroNLP/hateBERT")
    
    # Load the HateBERT model
    model = AutoModelForSequenceClassification.from_pretrained(
        "GroNLP/hateBERT",
        num_labels=num_labels,
        output_attentions=False,
        output_hidden_states=False
    )
    
    return model, tokenizer

# 3. Training function
def train_model(model, train_dataloader, test_dataloader, device, epochs=4):
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    
    # Learning rate scheduler
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training loop
    training_stats = []
    for epoch in range(epochs):
        print(f'======== Epoch {epoch + 1} / {epochs} ========')
        
        # Training
        model.train()
        total_train_loss = 0
        
        for batch in tqdm(train_dataloader):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Clear gradients
            model.zero_grad()
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_train_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update parameters
            optimizer.step()
            scheduler.step()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        
        # Testing/Evaluation
        model.eval()
        total_test_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in tqdm(test_dataloader):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # No gradient calculation needed
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
            
            loss = outputs.loss
            total_test_loss += loss.item()
            
            # Get predictions
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
        
        avg_test_loss = total_test_loss / len(test_dataloader)
        
        # Print epoch results
        print(f"  Average training loss: {avg_train_loss:.4f}")
        print(f"  Average test loss: {avg_test_loss:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds))
        
        # Save stats
        training_stats.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'test_loss': avg_test_loss
        })
    
    return model, training_stats

# 4. Main execution
def main():
    # Hyperparameters
    BATCH_SIZE = 16
    MAX_LENGTH = 128
    EPOCHS = 4
    RANDOM_SEED = 42
    
    # Set random seeds for reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Load dataset (replace with your dataset loading logic)
    # Example:
    # df = pd.read_csv('hate_speech_dataset.csv')
    # texts = df['text'].values
    # labels = df['label'].values  # 0: neutral, 1: implicit hate, 2: explicit hate
    
    # For demonstration purposes:
    texts = ["This is a neutral comment", "This group is less intelligent", "I hate this group"]
    labels = [0, 1, 2]  # neutral, implicit hate, explicit hate
    
    # Split data into train and test sets
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=RANDOM_SEED
    )
    
    # Initialize model and tokenizer
    model, tokenizer = initialize_model_and_tokenizer(num_labels=3)
    
    # Create datasets
    train_dataset = HateSpeechDataset(
        texts=train_texts,
        labels=train_labels,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )
    
    test_dataset = HateSpeechDataset(
        texts=test_texts,
        labels=test_labels,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Move model to device
    model = model.to(device)
    
    # Train model
    model, training_stats = train_model(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        device=device,
        epochs=EPOCHS
    )
    
    # Save the model
    model_save_path = 'finetuned_hatebert/'
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Visualize training results
    stats_df = pd.DataFrame(training_stats)
    plt.figure(figsize=(10, 6))
    plt.plot(stats_df['epoch'], stats_df['train_loss'], 'b-o', label='Training Loss')
    plt.plot(stats_df['epoch'], stats_df['test_loss'], 'r-o', label='Test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.show()

# 5. Inference function
def predict_hate_speech(text, model, tokenizer, device):
    # Set model to evaluation mode
    model.eval()
    
    # Tokenize input text
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
    
    # Map prediction to label
    label_map = {0: 'neutral', 1: 'implicit hate', 2: 'explicit hate'}
    
    return {
        'text': text,
        'prediction': label_map[prediction],
        'prediction_idx': prediction,
        'scores': torch.nn.functional.softmax(logits, dim=1).tolist()[0]
    }

if __name__ == "__main__":
    main()