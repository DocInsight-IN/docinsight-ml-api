import os
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from transformers import BertTokenizer, BertModel, BertConfig
from ml_deploy.topic_cluster.data.dataset import SiameseDataset

class SiameseBERT(nn.Module):
    def __init__(self, bert_model: BertModel):
        super(SiameseBERT, self).__init__()
        self.bert = bert_model
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.fc(pooled_output)
        return logits
    
def predict_batch(model, dataloader):
    predictions = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            outputs = model(input_ids, attention_mask)
            predictions.extend(outputs.squeeze().tolist())
    return predictions
    

def train_model(pairs, labels, tokenizer, siamese_model):
    train_pairs, val_pairs, train_labels, val_labels = train_test_split(pairs, labels, test_size=0.2, random_state=42)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(siamese_model.parameters(), lr=2e-5)

    train_dataset = SiameseDataset(train_pairs, train_labels, tokenizer, max_length=128)
    val_dataset = SiameseDataset(val_pairs, val_labels, tokenizer, max_length=128)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    num_epochs = 5
    for epoch in range(num_epochs):
        siamese_model.train()
        running_loss = 0.0
        for inputs, labels in train_dataloader:
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            labels = labels.unsqueeze(1)
            optimizer.zero_grad()
            outputs = siamese_model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_dataloader)
        print(f'Epoch {epoch+1}, Training Loss: {epoch_loss}')

    siamese_model.eval()
    val_losses = []
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            labels = labels.unsqueeze(1)
            outputs = siamese_model(input_ids, attention_mask)
            val_loss = criterion(outputs, labels)
            val_losses.append(val_loss.item())
    val_loss = sum(val_losses) / len(val_losses)
    print(f'Validation Loss: {val_loss}')


def save_model(tokenizer, model, model_save_dir):
    os.makedirs(model_save_dir, exist_ok=True)

    # save the model state dict
    model_state_dict_path = os.path.join(model_save_dir, "siamese_bert_model_state_dict.pth")
    torch.save(model.state_dict(), model_state_dict_path)

    # save the model config
    model_config_path = os.path.join(model_save_dir, "siamese_bert_model_config.json")
    with open(model_config_path, 'w') as f:
        f.write(model.bert.config.to_json_string())

    tokenizer_save_path = os.path.join(model_save_dir, "tokenizer_config.json")
    tokenizer.save_pretrained(tokenizer_save_path)

def load_model(model_save_dir):
    tokenizer = BertTokenizer.from_pretrained(model_save_dir)
    model_config_path = os.path.join(model_save_dir, "siamese_bert_model_config.json")
    model_config = BertConfig.from_json_file(model_config_path)

    model_state_dict_path = os.path.join(model_save_dir, "siamese_bert_model_state_dict.pth")
    model = SiameseBERT(BertModel(model_config))
    model.load_state_dict(torch.load(model_state_dict_path))

    return tokenizer, model