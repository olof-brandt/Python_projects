import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import os
import random


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


def load_test_data(training_data_dir='training_data', num_samples=5):
    import glob
    
    # Find all files matching jobcards_yymmdd.json pattern
    pattern = os.path.join(training_data_dir, 'jobcards_*.json')
    json_files = glob.glob(pattern)
    
    if not json_files:
        raise FileNotFoundError(f'No files matching jobcards_yymmdd.json found in {training_data_dir}')
    
    print(f'Found {len(json_files)} training data files:')
    for f in json_files:
        print(f'  - {os.path.basename(f)}')
    
    all_samples = []
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            all_samples.append({
                'text': item['text'],
                'label': item['label'],
                'label_id': item['label_id'],
                'sub_label': item.get('sub_label', ''),
                'sub_labels': item.get('sub_labels', [])
            })
    
    # Randomly sample num_samples
    if len(all_samples) > num_samples:
        sampled = random.sample(all_samples, num_samples)
    else:
        sampled = all_samples
    
    return sampled


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
    TESTING_DATA_DIR = 'testing_data'
    MODEL_NAME = 'KB/bert-base-swedish-cased'
    MAX_LENGTH = 512
    MODEL_SAVE_PATH = 'jobcards_model/jobcard_classifier.pth'
    NUM_TEST_SAMPLES = 5
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load tokenizer
    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load model
    print('Loading model...')
    model = JobCardClassifier(n_classes=2, model_name=MODEL_NAME)
    
    # Check if model file exists
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f'Error: Model file not found at {MODEL_SAVE_PATH}')
        print('Please train the model first using jobcards_nn_training_261017.py')
        return
    
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model = model.to(device)
    print('Model loaded successfully')
    
    # Load test data
    print(f'\nLoading test data (sampling {NUM_TEST_SAMPLES} items)...')
    test_samples = load_test_data(TESTING_DATA_DIR, NUM_TEST_SAMPLES)
    print(f'Sampled {len(test_samples)} test items\n')
    
    # Label mapping
    label_map = {0: 'no', 1: 'yes'}
    
    # Test each sample
    print('=' * 80)
    print('TESTING JOB CARD CLASSIFIER')
    print('=' * 80)
    
    correct = 0
    total = len(test_samples)
    
    for i, sample in enumerate(test_samples, 1):
        text = sample['text']
        actual_label = sample['label']
        actual_label_id = sample['label_id']
        sub_label = sample['sub_label']
        sub_labels = sample['sub_labels']
        
        # Get prediction
        pred_id, probs = predict_single(model, tokenizer, text, device, MAX_LENGTH)
        pred_label = label_map[pred_id]
        
        # Check if correct
        is_correct = pred_id == actual_label_id
        if is_correct:
            correct += 1
        
        # Print results
        print(f'\n--- Test Sample {i} ---')
        print(f'Text (first 200 chars): {text[:200]}...')
        print(f'Predicted: {pred_label.upper()} (confidence: {probs[pred_id]:.4f})')
        print(f'Actual: {actual_label.upper()} (label_id: {actual_label_id})')
        print(f'Correct: {is_correct}')
        if sub_label:
            print(f'Actual sub-label: {sub_label}')
        if sub_labels:
            print(f'Actual sub-labels: {", ".join(sub_labels)}')
        print(f'Probabilities - No: {probs[0]:.4f}, Yes: {probs[1]:.4f}')
    
    # Summary
    print('\n' + '=' * 80)
    print('SUMMARY')
    print('=' * 80)
    print(f'Total samples tested: {total}')
    print(f'Correct predictions: {correct}')
    print(f'Accuracy: {correct/total:.2%}')


if __name__ == '__main__':
    main()
