from pathlib import Path
import torch 
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device=torch.device(
    'cuda' if torch.cuda.is_available() else'cpu'
)

project_dir=Path(__file__).resolve().parent.parent
model_dir=project_dir/'bert_clinc_model'

tokenizer=AutoTokenizer.from_pretrained(model_dir)

model=AutoModelForSequenceClassification.from_pretrained(
    model_dir
)

model=model.to(device)
model.eval()

def inference(text):
    inputs=tokenizer(
        text,
        truncation=True,
        padding='max_length',
        return_tensors='pt',
        max_length=128
    )
    inputs={
        key:value.to(device)
        for key,value in inputs.items()
    }
    with torch.no_grad():
        outputs=model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
    probabilities=torch.softmax(
        outputs.logits,
        dim=-1
    )
    prediction=torch.argmax(
        probabilities,
        dim=1
    ).item()

    confidence=probabilities[0][prediction].item()
    label=model.config.id2label[prediction]
    return label, confidence

#==================
#Test inference
#==================

while True:
    text=input('Enter customer message:')
    if text.lower()=='exit':
        print('exiting....')
        break
    label,confidence=inference(text)
    print('Intent:',label)
    print(f'Confidence: {confidence *100:.2f}%')
    print()
    
