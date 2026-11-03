import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm import tqdm
import os


class JobCardDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class JobCardClassifier(nn.Module):
    def __init__(self, n_classes, model_name='KB/bert-base-swedish-cased'):
        super(JobCardClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        return self.fc(output)


def load_training_data(training_data_dir='training_data'):
    import glob
    
    # Find all files matching jobcards_yymmdd.json pattern
    pattern = os.path.join(training_data_dir, 'jobcards_*.json')
    json_files = glob.glob(pattern)
    
    if not json_files:
        raise FileNotFoundError(f'No files matching jobcards_yymmdd.json found in {training_data_dir}')
    
    print(f'Found {len(json_files)} training data files:')
    for f in json_files:
        print(f'  - {os.path.basename(f)}')
    
    all_texts = []
    all_labels = []
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = [item['text'] for item in data]
        labels = [item['label_id'] for item in data]
        
        all_texts.extend(texts)
        all_labels.extend(labels)
    
    return all_texts, all_labels


def train_epoch(model, data_loader, optimizer, device, scheduler, class_weights=None):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    if class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    for batch in tqdm(data_loader, desc='Training'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels)
        total_samples += labels.size(0)

    return total_loss / len(data_loader), correct_predictions.double() / total_samples


def eval_model(model, data_loader, device, class_weights=None):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    if class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_samples += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(data_loader), correct_predictions.double() / total_samples, all_preds, all_labels


def predict_single(model, tokenizer, text, device, max_length=512):
    model.eval()
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_length,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, prediction = torch.max(outputs, dim=1)
        probabilities = torch.softmax(outputs, dim=1)
    
    return prediction.item(), probabilities.cpu().numpy()[0]


def main():
    # Configuration
    TRAINING_DATA_DIR = 'training_data'  # Directory containing jobcards_yymmdd.json files
    MODEL_NAME = 'KB/bert-base-swedish-cased'  # Swedish BERT model
    MAX_LENGTH = 512
    BATCH_SIZE = 8
    EPOCHS = 5
    LEARNING_RATE = 2e-5
    MODEL_SAVE_PATH = 'jobcards_model/jobcard_classifier.pth'
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create model directory if it doesn't exist
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    # Load data
    print('Loading training data...')
    texts, labels = load_training_data(TRAINING_DATA_DIR)
    print(f'Loaded {len(texts)} samples')
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f'Train samples: {len(train_texts)}, Validation samples: {len(val_texts)}')
    
    # Calculate class weights to handle imbalance
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    print(f'Class weights: {class_weights}')
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Datasets
    train_dataset = JobCardDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = JobCardDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    model = JobCardClassifier(n_classes=2, model_name=MODEL_NAME)
    model = model.to(device)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    
    # Training loop
    best_accuracy = 0
    print('\nStarting training...')
    
    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch + 1}/{EPOCHS}')
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, scheduler, class_weights)
        print(f'Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}')
        
        val_loss, val_acc, val_preds, val_labels = eval_model(model, val_loader, device, class_weights)
        print(f'Val loss: {val_loss:.4f}, Val accuracy: {val_acc:.4f}')
        
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f'Saved best model with accuracy: {best_accuracy:.4f}')
    
    print(f'\nTraining complete. Best validation accuracy: {best_accuracy:.4f}')
    
    # Load best model for inference
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model = model.to(device)
    
    # Example inference
    print('\nExample inference:')
    sample_text = "Elektronikkonstruktör sökes till innovativt företag i Stockholm"
    pred, probs = predict_single(model, tokenizer, sample_text, device)
    label_map = {0: 'no', 1: 'yes'}
    print(f'Text: {sample_text}')
    print(f'Prediction: {label_map[pred]}')
    print(f'Probabilities: No={probs[0]:.4f}, Yes={probs[1]:.4f}')


if __name__ == '__main__':
    main()
