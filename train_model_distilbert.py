# 0. FORCE CPU BEFORE ANY TORCH IMPORT
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # disable GPU completely
os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # optional safety

# 1. Imports
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# 2. Device
device = torch.device("cpu")
print("Training on device:", device)

# 3. Load dataset
fake_df = pd.read_csv('Fake.csv')
true_df = pd.read_csv('True.csv')

fake_df['label'] = 0
true_df['label'] = 1

df = pd.concat([fake_df[['text','label']], true_df[['text','label']]], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Optional: smaller dataset for CPU-friendly training
df = df.sample(n=min(len(df), 1000), random_state=42)

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# 4. Tokenizer & Dataset
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

class NewsDataset(Dataset):
    def __init__(self, texts, labels, max_length=128):
        self.encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=max_length)
        self.labels = list(labels)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = NewsDataset(X_train, y_train)
test_dataset = NewsDataset(X_test, y_test)

# 5. Model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
model.to(device)

# 6. Training Arguments (CPU-only)
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    logging_dir='./logs',
    logging_steps=10,
    no_cuda=True,  # FORCE CPU
    save_strategy='no',
)

# 7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# 8. Train
trainer.train()

# 9. Save model & tokenizer
os.makedirs('distilbert_model', exist_ok=True)
model.save_pretrained('distilbert_model')
tokenizer.save_pretrained('distilbert_model')

print("DistilBERT CPU-trained model saved successfully!")


