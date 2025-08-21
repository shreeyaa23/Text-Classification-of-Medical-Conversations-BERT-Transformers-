# Text Classification of Medical Conversations  

## Project Overview  
This project develops a **text classification system** to categorize medical conversations into **15 intent categories** (e.g., GREET, THANK, REQUEST_INFORMATION, DIAGNOSE) based on **Social Support Theory**.  
The objective is to enhance **clarity, patient satisfaction, and healthcare outcomes** by applying state-of-the-art machine learning and transformer models.  

---

## Dataset  
- **Size**: 4,030 labeled sentences  
- **Domain**: Medical conversation transcripts  
- **Categories**: Emotional Support (e.g., CONSOLE, FUTURE_SUPPORT) and Informational Support (e.g., DIAGNOSE, TREAT, REQUEST_INFORMATION)  

---

## Methodology  

1. **Data Preparation**  
   - Preprocessing (cleaning, tokenization, augmentation)  
   - Imbalance handling using weighted cross-entropy and focal loss  

2. **Feature Engineering**  
   - TF-IDF vectors for traditional ML models  
   - FastText embeddings for CNN and LSTM models  
   - Contextual embeddings from **BERT tokenizer** for transformer models  

3. **Model Development**  
   - Traditional ML: Logistic Regression, SVM, Random Forest  
   - Deep Learning: CNN, Bi-LSTM  
   - Transformers: **DistilBERT, FreezeBERT, DeBERTa, BioClinicalBERT**  

4. **Evaluation**  
   - Metrics: Accuracy, Macro F1-score, Precision/Recall  
   - Hyperparameter tuning and early stopping to prevent overfitting  

---

## Results  

| Model                  | Accuracy | Macro F1 |
|-------------------------|----------|----------|
| Logistic Regression     | 67.65%   | –        |
| SVM                     | 64.87%   | –        |
| Random Forest           | 72.37%   | –        |
| CNN                     | 71.04%   | –        |
| Bi-LSTM                 | 73.57%   | –        |
| **DeBERTa-v3-large**    | **83.57%** | **85%** |

**Key Findings**  
- Transformer-based models significantly outperformed traditional ML and deep learning baselines.  
- **DeBERTa-v3-large** achieved the best results, demonstrating the effectiveness of contextual embeddings for medical intent classification.  
- Handling class imbalance (focal loss, weighted loss) was critical to achieving balanced performance.  

---

## How to Run  

### 1. Clone Repository  
```bash
git clone https://github.com/your-username/medical-conversation-classification.git
cd medical-conversation-classification

### 2. Install Dependencies
pip install -r requirements.txt

### 3. Train Model
python train.py --model deberta

### 4. Evaluate Model
python evaluate.py --model deberta

## Tools & Technologies

Programming: Python (pandas, numpy, scikit-learn)
Deep Learning: PyTorch / TensorFlow
NLP: Hugging Face Transformers, FastText
Visualization: Matplotlib, Seaborn

## Future Enhancements

Extend dataset with multi-lingual conversations
Experiment with large language models (LLMs) for zero-shot and few-shot classification
Deploy classification system as an API for real-time healthcare applications
