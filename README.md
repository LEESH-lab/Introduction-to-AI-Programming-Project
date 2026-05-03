# Introduction-to-AI-Programming-Project
# Fashion Product Image Classification: Solving the Long-Tail Problem

## 1. Project Goal
The primary objective of this project is to develop a robust image classification model for fashion products while addressing the extreme **Long-Tail data imbalance problem**. Using EfficientNetV2-S as the baseline, we implemented a **Class-Balanced Loss** to improve the recall of minority classes (rare items) without heavily compromising the performance of majority classes. The goal is to build a fair classification system applicable to real-world e-commerce scenarios.

## 2. Dataset Information
* **Dataset:** [Ashraq/Fashion Product Images Small](https://huggingface.co/datasets/ashraq/fashion-product-images-small) (Hugging Face Datasets)
* **Classes:** [본인의 클래스 개수 입력, 예: 142] categories of fashion products
* **Resolution:** 60x80 pixels (RGB)
* **Distribution:** Highly imbalanced long-tail distribution (Majority classes like 'Tshirts' dominate, while minority classes have very few samples).

## 3. Environment and Dependencies
To run this project, you need the following environment. 
* **OS:** [예: Ubuntu 20.04 / Windows 10]
* **Hardware:** [예: NVIDIA RTX 3090 / Google Colab T4 GPU]
* **Python Version:** Python 3.8+

Install the required dependencies using the following command:
```bash
pip install torch torchvision datasets numpy matplotlib pandas
