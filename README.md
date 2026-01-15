# 😵‍💫 HallucinoGenAI: AI Hallucination Detection System

**A State-of-the-Art (SOTA) NLP system designed to detect "hallucinations" in Large Language Models (LLMs) by verifying logical consistency between knowledge bases and AI responses.**

---

## 📌 Business Context & Problem Statement

### **The Problem: The "Confident Liar"**
Generative AI models (like GPT-4, Claude, Gemini) have a critical flaw: **Hallucinations**. They can confidently state factually incorrect information.
* **Business Risk:** In high-stakes industries like **Finance, Legal, and Healthcare**, a single hallucination can lead to:
    * **Regulatory Fines:** Misquoting a statute or financial regulation.
    * **Liability:** Providing incorrect medical or safety advice.
    * **Erosion of Trust:** Users lose faith in the automated system.

### **The Solution: HallucinoGenAI**
We cannot rely on an LLM to police itself (self-correction often fails). **HallucinoGenAI** acts as an external, specialized **"Judge Model"**. It is a fine-tuned Cross-Encoder that takes two inputs—a **Premise** (Source of Truth) and a **Hypothesis** (AI Response)—and classifies the response into one of three safety categories:
1.  **Factual (Entailment):** Safe to deploy.
2.  **Irrelevant (Neutral):** Off-topic but harmless.
3.  **Hallucination (Contradiction):** **CRITICAL RISK** - Must be blocked.

---

## 🛠️ Technical Architecture

### **1. Model Selection: Why DeBERTa?**
We moved beyond standard BERT models to use **Microsoft's DeBERTa-v3-small (Cross-Encoder)**.
* **Disentangled Attention:** Unlike BERT, DeBERTa uses a disentangled attention mechanism that represents words and their positions separately. This allows it to understand *nuance* better than larger models.
* **NLI Optimization:** The model was pre-trained specifically on Natural Language Inference (NLI) tasks, making it the current SOTA for logic detection in the "small model" weight class.

### **2. The "Hard Negative" Data Strategy**
Standard datasets are too easy. A model can guess that *"The sky is blue"* contradicts *"I love pizza"* just by looking at keywords.
* **Our Innovation:** We engineered **"Hard Negatives"**—synthetic examples where the text is 99% identical, but one critical fact is changed (e.g., changing a date from *1969* to *1970*).
* **Result:** This forces the model to learn **semantic logic** rather than just keyword matching.

### **3. Weighted Loss Function (Safety First)**
In this domain, **False Negatives (Missing a Lie)** are 100x worse than False Positives (Flagging a Fact).
* **Technical Implementation:** We implemented a custom `Trainer` with a **Weighted Cross-Entropy Loss**.
* **The Weighting:** We penalized the model **2.0x** more for missing a Hallucination than for other errors.
    ```python
    # Penalize Hallucination errors (Index 2) double
    weights = torch.tensor([1.0, 1.0, 2.0])
    loss_fct = nn.CrossEntropyLoss(weight=weights)
    ```
---

## 📊 Key Insights & Results

### **1. Threshold Optimization (ROC Curve)**
We discovered that the standard probability threshold (50%) was too risky for production.
* **Insight:** To achieve **95% Recall** (catching 95% of lies), we lowered the decision threshold to **~30%**.
* **Action:** Any response where the model is even 30% confident it's a lie is automatically flagged for human review.

### **2. Explainability (The "Why")**
Trust requires transparency. We built a **Perturbation-Based Explainability Module**.
* **How it works:** We mask individual words in the AI response and measure the drop in the "Hallucination Score."
* **Result:** A heatmap that highlights exactly *which word* (e.g., a specific date or name) triggered the alarm.

---

## 🚀 How to Run

### **Prerequisites**
```bash
pip install transformers datasets torch scikit-learn seaborn matplotlib sentence-transformers
```

### **Quick Start**

The notebook `HallucinoGenAI_Hallucination_Detection.ipynb` is self-contained. It will:

1. Download the **DeBERTa** model.
2. Generate the **Hard Negative** dataset.
3. Train with **Weighted Loss**.
4. Output the **Confusion Matrix** and **Risk Heatmap**.

---

## 🏆 Project Achievements

* **Accuracy:** Achieved **67%+ Accuracy** on the difficult NLI task (beating baseline logic).
* **Safety:** Optimized for **95% Recall** on Hallucinations via Threshold Tuning.
* **Innovation:** Implemented **Hard Negative Mining** and **Weighted Loss Training** to solve the "Imbalanced Risk" problem.

---

## 👤 Author

* **Shagun Sharma** - *Machine Learning Engineer*

**Graduate Student, Duke University - Fuqua School of Business**

  * [GitHub Profile](https://github.com/Shaguns26)


