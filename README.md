# Fashion Product Image Classification: Solving the Long-Tail Problem

**Team Members:** Donggyu Yu (20230349), Jisu Kim (20220561), Seunghyun Lee (20220422)

## 1. Project Goal
The primary objective of this project is to design and evaluate a CNN model capable of accurately classifying various categories of fashion products. In the fashion e-commerce domain, a significant practical challenge is the extreme **"Class Imbalance" (Long-tail) problem**. 
To overcome this, we adopted the parameter-efficient **EfficientNet-V2-S** architecture as the baseline and introduced a **Class-Weighted Loss** function alongside **Custom Data Augmentation** techniques to mitigate majority class bias and successfully improve the recall of rare minority classes.

## 2. Dataset Information
* **Dataset:** [Fashion Product Images (Small)](https://huggingface.co/datasets/ashraq/fashion-product-images-small)
* **Number of Classes:** 141 unique fashion product categories.
* **Resolution:** 60x80 pixels (RGB).
* **Characteristics:** Extreme long-tail distribution.

## 3. Environment and Dependencies
Ensure you have the following environment set up:
* **Python Version:** 3.8+
* **Framework:** PyTorch
* **Hardware:** GPU recommended

Install the required dependencies using pip:
```bash
pip install torch torchvision datasets numpy matplotlib pandas
```

## 4. Training Instructions
To train the EfficientNet-V2-S model with Class-Weighted Loss and Custom Data Augmentation, run the `train.py` script. The script automatically downloads the Hugging Face dataset and saves the weights as `best_model.pth`.
```bash
python train.py --batch_size 32 --epochs 20 --lr 0.001
```

## 5. Evaluation Instructions
To evaluate the trained model on the test dataset and generate visual performance metrics (such as the 141x141 Confusion Matrix, Minority/Majority class graphs), run:
```bash
python evaluate.py --model_path ./best_model.pth
```

## 6. How to Run the Inference Demo
We provide an inference script to test the model on individual images. You can pass a sample image to see the model's top predicted fashion category and its confidence score.
```bash
python inference_demo.py --model_path ./best_model.pth --image_path ./sample_image.jpg
```
