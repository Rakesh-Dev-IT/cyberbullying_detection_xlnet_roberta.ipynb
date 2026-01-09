# Hybrid Cyberbullying Detection using XLNet and RoBERTa

## Overview
With the rapid growth of social media platforms, online harassment and cyberbullying have become serious concerns. Manual moderation techniques such as reporting and blocking are often ineffective at scale.  
This project presents an **automatic cyberbullying detection system** using a **hybrid deep learning approach** that combines **XLNet** and **RoBERTa** transformer models to accurately classify harmful textual content.

---

## Problem Statement
Social media users are frequently exposed to abusive, hateful, and offensive language. Traditional machine learning models and single-transformer approaches often fail to capture both:
- Global sentence structure  
- Fine-grained semantic meaning  

This project addresses these limitations by leveraging the complementary strengths of **multiple transformer architectures**.

---

## Proposed Methodology
A **hybrid transformer-based architecture** is proposed using **XLNet** and **RoBERTa**.

- A balanced cyberbullying dataset collected from multiple online sources is used
- Data is split into **training and validation sets (90:10, stratified)**
- Dual contextual embeddings are generated using XLNet and RoBERTa tokenizers
- A custom PyTorch `Dataset` class processes inputs in parallel
- CLS representations from both models are concatenated
- The combined embedding is passed through a fully connected neural network for **binary classification**

---

## Model Architecture
- **XLNet** – permutation-based language modeling for enhanced bidirectional context  
- **RoBERTa** – robustly optimized BERT with dynamic masking  
- **Concatenation** of pooled CLS embeddings  
- **Fully connected classification layer**

---

## Training Details
- **Framework:** PyTorch  
- **Optimizer:** AdamW  
- **Learning Rate Scheduler:** Cosine schedule with warmup  
- **Loss Function:** CrossEntropyLoss  
- **Epochs:** 3  
- **Evaluation Metrics:** Accuracy, F1-score, Confusion Matrix  

---

## Results and Analysis
The hybrid **XLNet–RoBERTa** model achieved a significantly higher **F1-score** compared to:
- Traditional machine learning models (Logistic Regression, Linear SVC)  
- Single-transformer baselines  

The combined architecture captures both **global contextual relationships** and **local semantic information**, leading to improved performance while maintaining efficient training time.

---

## Key Advantages
- Eliminates extensive manual feature engineering  
- Leverages transfer learning from multiple pre-trained models  
- Outperforms traditional and single-model approaches  
- Scalable and suitable for real-world deployment  

---

## Deployment
The trained model is deployed using **Gradio**, providing:
- Real-time text input  
- Color-coded prediction output  
- User-friendly web interface  

---

## Future Work
- Extend the architecture with additional transformer models  
- Evaluate on larger and multilingual datasets  
- Optimize inference speed for production deployment  
- Integrate advanced moderation dashboards  

---

## How to Run
```bash
pip install -r requirements.txt
python src/train.py
python src/evaluate.py
python src/app.py
