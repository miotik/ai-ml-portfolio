from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
import torch 
from pathlib import Path

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = load_dataset("DeepPavlov/clinc_oos", "plus")
tokenizer=AutoTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=128
    )

tokenized_dataset=dataset.map(
    tokenize_function,
    batched=True
)

tokenized_dataset.set_format(
    type='torch',
    columns=['input_ids','attention_mask','label']
)

train_loader = DataLoader(
    tokenized_dataset["train"],
    batch_size=16,
    shuffle=True
)

validation_loader = DataLoader(
    tokenized_dataset["validation"],
    batch_size=16
)

test_loader = DataLoader(
    tokenized_dataset["test"],
    batch_size=16
)

model=AutoModelForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=151
)

model = model.to(device)
optimizer=torch.optim.AdamW(
    model.parameters(),
    lr=2e-5
)

num_epochs = 2
for epoch in range(num_epochs):

    # --------------------
    # Training
    # --------------------
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["label"]
        )
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    average_train_loss = total_train_loss / len(train_loader)

    # --------------------
    # Validation
    # --------------------
    model.eval()
    total_val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in validation_loader:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["label"]
            )
            total_val_loss += outputs.loss.item()
            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == batch["label"]).sum().item()
            total += batch["label"].size(0)

    average_val_loss = total_val_loss / len(validation_loader)
    validation_accuracy = correct / total
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Training loss: {average_train_loss:.4f}")
    print(f"Validation loss: {average_val_loss:.4f}")
    print(f"Validation accuracy: {validation_accuracy:.4f}")
    print()

#====================
#Testing
#====================

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        predictions = torch.argmax(outputs.logits, dim=1)
        correct += (predictions == batch["label"]).sum().item()
        total += batch["label"].size(0)
test_accuracy = correct / total
print("Test accuracy:", test_accuracy)

#=======================
#Inference
#=======================

label_names ={}
for label,label_text in zip(
    dataset['train']['label'],
    dataset['train']['label_text']
):
    label_names[label]=label_text

def inference(text):
    inputs=tokenizer(
        text,
        truncation=True,
        return_tensors='pt',
        padding='max_length',
        max_length=128
    )
    inputs={
        key:value.to(device)
        for key,value in inputs.items()
    }
    model.eval()
    with torch.no_grad():
        outputs=model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
    logits=outputs.logits
    probabilities=torch.softmax(logits,dim=1)
    prediction=torch.argmax(probabilities, dim=1).item()
    confidence=probabilities[0][prediction].item()
    label=label_names[prediction]
    return label,confidence

test_inputs = [
    "I want to check my account balance",
    "My card was declined",
    "I forgot my PIN",
    "I want to transfer money",
    "What currencies do you support?"
]

for text in test_inputs:

    label, confidence = inference(text)

    print("Text:", text)
    print("Intent:", label)
    print(f"Confidence: {confidence * 100:.2f}%")
    print()

# --------------------
# Save trained model
# --------------------

project_dir=Path(__file__).resolve().parent.parent
model_dir=project_dir/'bert_clinc_model'

model.config.id2label={
    i:label_names[i]
    for i in label_names
}

model.config.label2id={
    label:i
    for i, label in model.config.id2label.items()
}

model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)

print(f'Model and tokenizer saved to: {model_dir}')