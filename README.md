# Hybrid Cyberbullying Detection using XLNet and RoBERTa

## Overview

Information and Communication Technologies have transformed social interaction but have also increased online threats such as cyberbullying. Manual moderation techniques such as reporting or blocking are often insufficient to handle the scale of harmful content. This project presents an automatic cyberbullying detection system using a hybrid deep learning approach that combines XLNet and RoBERTa transformer models.

## Problem Statement

Social media platforms expose users to harmful and abusive content at scale. Traditional machine learning techniques and single-transformer models often struggle to capture both global contextual dependencies and fine-grained semantic information. This project addresses these limitations by leveraging the complementary strengths of multiple transformer architectures.

## Proposed Methodology

A hybrid transformer-based architecture combining XLNet and RoBERTa is proposed. A balanced cyberbullying dataset collected from multiple online sources was used and stratified across various cyberbullying categories. The dataset was split into training and validation sets using a 90:10 stratified ratio.

Dual contextual embeddings were generated using XLNet and RoBERTa tokenizers. A custom PyTorch Dataset class was implemented to process inputs in parallel. The pooled CLS representations from both transformer models were concatenated to form a joint embedding space, which was passed through a fully connected neural network for multi-class classification.

## Model Architecture

* XLNet (permutation-based language modeling for enhanced bidirectional context)
* RoBERTa (robustly optimized BERT with dynamic masking)
* Concatenation of pooled CLS embeddings
* Fully connected classification layer

## Training Details

* Framework: PyTorch
* Optimizer: AdamW
* Learning Rate Scheduler: Cosine schedule with warmup
* Loss Function: CrossEntropyLoss
* Epochs: 3
* Evaluation Metrics: Accuracy, F1-score, Training efficiency

## Results and Analysis

The hybrid XLNet–RoBERTa model achieved a significantly higher F1-score compared to traditional machine learning models and single-transformer baselines. While XLNet and RoBERTa individually performed well, their combination produced a synergistic effect by capturing both global sentence structure and local semantic information.

The hybrid model outperformed Linear SVC and Logistic Regression models trained on engineered features, while maintaining efficient training time and scalability for high-dimensional textual data.

## Key Advantages

* Eliminates extensive manual feature engineering
* Leverages transfer learning from multiple pre-trained transformer models
* Improved performance over single-model and traditional approaches
* Scalable and suitable for real-world applications

## Future Work

* Extend the architecture to include additional transformer models
* Deploy the system using interactive interfaces such as Gradio
* Evaluate performance on larger and multilingual datasets

## Author

Muthyala Rakesh
