# BERT Customer Support Intent Classification System

A customer-support NLP system that uses **BERT to understand a customer's message and identify their intent**.

For example:

```text
Input:
"My card was stolen"

Prediction:
report_lost_card

Confidence:
58.99%
```

The project covers the complete process of taking an NLP model from **training → inference → API → Docker deployment**.

---

## 1. What Problem Does This Solve?

Customer-support platforms receive thousands of messages such as:

```text
"My card was stolen"
"I forgot my PIN"
"Why is my transfer pending?"
"Where is the nearest ATM?"
```

Before a support system can respond appropriately, it needs to understand **what the customer is asking about**.

This project uses BERT to classify each message into one of **151 predefined customer-support intents**.

The predicted intent can then be used by a larger customer-support system to decide what should happen next.

---

## 2. How the System Works

The overall pipeline is:

```text
Customer Message
       ↓
    Tokenizer
       ↓
 Fine-tuned BERT
       ↓
 Intent Prediction
       ↓
 Confidence Score
       ↓
     FastAPI
       ↓
     Docker
```

For example:

```text
"My card was stolen"
        ↓
BERT
        ↓
"report_lost_card"
        ↓
0.5899 confidence
```

---

## 3. Dataset

The model was trained using the **CLINC-OOS** dataset from Hugging Face.

Dataset:

```text
DeepPavlov/clinc_oos
```

The project uses the `plus` configuration.

The dataset contains customer-support-style queries covering many different intents.

The model was trained to classify:

**151 different intents**

Examples include:

```text
report_lost_card
pin_change
balance
cash_withdrawal
card_payment_fee_charged
transfer_pending
```

---

## 4. Model

The base model used is:

```text
bert-base-uncased
```

I fine-tuned BERT for sequence classification with **151 output classes**.

Conceptually:

```text
Input Text
    ↓
BERT
    ↓
Classification Layer
    ↓
151 class scores
    ↓
Predicted Intent
```

The tokenizer uses a maximum sequence length of **128 tokens**.

---

## 5. Training

The main training configuration was:

| Setting                 | Value             |
| ----------------------- | ----------------- |
| Base model              | BERT Base Uncased |
| Number of classes       | 151               |
| Optimizer               | AdamW             |
| Learning rate           | 2e-5              |
| Batch size              | 16                |
| Epochs                  | 2                 |
| Maximum sequence length | 128               |

The trained model and tokenizer are saved locally so they can be loaded later for inference.

---

## 6. Model Performance

The model achieved approximately:

```text
Validation Accuracy: ~94.3%
Test Accuracy:       ~87.8%
```

The exact result can vary slightly because the training run was not performed with a fixed random seed.

Accuracy was not the only thing I considered. In a real customer-support system, incorrect classifications and uncertain predictions are also important, so confidence handling and future out-of-distribution detection are areas for improvement.

---

## 7. Inference

After training, I created a separate inference pipeline.

The process is:

```text
New Customer Message
        ↓
Tokenization
        ↓
BERT
        ↓
Logits
        ↓
Softmax
        ↓
Highest Probability
        ↓
Intent + Confidence
```

For example:

```text
Input:
"My card was stolen"

Output:
Intent: report_lost_card
Confidence: 58.99%
```

The model's `id2label` mapping converts the predicted class number into the corresponding intent name.

---

## 8. FastAPI

The trained model is exposed through a REST API using **FastAPI**.

### Endpoint

```text
POST /predict
```

### Request

```json
{
  "text": "My card was stolen"
}
```

### Response

```json
{
  "intent": "report_lost_card",
  "confidence": 0.5899108052253723
}
```

FastAPI also provides interactive Swagger documentation:

```text
/docs
```

This makes it easy to test the model through a browser or connect it to another application.

---

## 9. Docker

The API and trained model are packaged into a Docker container.

The Docker image contains:

```text
Python
PyTorch
Transformers
FastAPI
Uvicorn
Source Code
Trained BERT Model
```

The application can be started with:

```bash
docker build -t bert-intent-api .
```

and:

```bash
docker run -p 8000:8000 bert-intent-api
```

This makes the application reproducible and avoids depending on the exact local Python environment.

---

## 10. Public API Testing

For development and demonstration, I used **ngrok** to expose the locally running Docker API through a temporary public URL.

The architecture becomes:

```text
Internet
    ↓
  ngrok
    ↓
Docker Container
    ↓
 FastAPI
    ↓
   BERT
```

This allowed the API to be tested from outside the local machine without paying for cloud hosting.

---

## 11. Project Structure

```text
BERT_Intent_Classification_System/
│
├── Dockerfile
├── README.md
├── requirements.txt
│
├── api/
│   └── __init__.py
│
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── dataset.py
│   └── inference.py
│
└── bert_clinc_model/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── tokenizer_config.json
```

### Important files

**`dataset.py`**

Handles dataset loading, label mapping, tokenization, training, evaluation, and saving the trained model.

**`inference.py`**

Loads the trained model and provides a reusable inference function that returns:

```text
intent
confidence
```

**`api.py`**

Connects the inference function to FastAPI.

**`Dockerfile`**

Packages the application and model into a reproducible Docker image.

---

## 12. Limitations

The current model is a **closed-set classifier**.

This means it knows only the 151 intents it was trained on.

If a customer asks something completely unrelated, the model may still choose one of those 151 classes.

For example:

### Request

```json
{
  "text": "i want to order a pizza."
}
```

### Response

```json
{
  "intent": "order",
  "confidence": 0.4750872552394867
}
```

The model could still return one of the available banking-related intents.

The confidence score also does not guarantee that a prediction is correct.

---

## 13. Future Improvements

The next stage of this project is to turn the classifier into a more complete customer-support system.

Planned improvements include:

### Better uncertainty handling

Add a confidence threshold so that uncertain predictions can be routed to another process instead of blindly accepting the classification.

```text
High confidence
      ↓
Use predicted intent

Low confidence
      ↓
Ask for clarification / use fallback
```

### LLM integration

Use the predicted intent to route the customer's request to an LLM that can generate a natural-language response.

```text
Customer Message
       ↓
BERT Intent Classifier
       ↓
Predicted Intent
       ↓
Relevant Context
       ↓
LLM
       ↓
Customer Response
```

### RAG

Add retrieval from a customer-support knowledge base so that the LLM can generate responses based on relevant company information rather than relying only on its internal knowledge.

---

## 14. Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* BERT
* FastAPI
* Pydantic
* Docker
* Git
* Git LFS
* ngrok

---

## 15. Key Takeaway

This project was built to demonstrate more than simply training a BERT model.

It covers the complete path:

```text
Dataset
   ↓
Data Processing
   ↓
BERT Fine-Tuning
   ↓
Evaluation
   ↓
Model Saving
   ↓
Inference
   ↓
FastAPI
   ↓
Docker
   ↓
Public API Testing
```

The current system provides the **intent-classification layer** of a larger customer-support application, with LLM-based response generation and RAG planned as the next stages.
