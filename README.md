# Fashion Product Image Classification: Solving the Long-Tail Problem

**Team Members:** Donggyu Yu (20230349), Jisu Kim (20220561), Seunghyun Lee (20220422)

## 1. Project Goal
The primary objective of this project is to design and evaluate a CNN model capable of accurately classifying various categories of fashion products. In the fashion e-commerce domain, a significant practical challenge is the extreme **"Class Imbalance" (Long-tail) problem**. While popular items (e.g., T-shirts) have abundant data, specialized items (e.g., iPads, duffle bags) severely lack training data. 

To overcome this, we adopted the parameter-efficient **EfficientNet-V2-S** architecture as the baseline and introduced a **Class-Weighted Loss** function alongside **Custom Data Augmentation** techniques. The ultimate goal is to mitigate majority class bias and successfully improve the recall of rare minority classes, creating a fair and balanced classification system.

## 2. Dataset Information
* **Dataset:** [Fashion Product Images (Small)](https://huggingface.co/datasets/ashraq/fashion-product-images-small) (Provided by Ashraq on Hugging Face)
* **Number of Classes:** 141 unique fashion product categories.
* **Resolution:** 60x80 pixels (RGB).
* **Characteristics:** Highly imbalanced long-tail distribution where predictions in standard models are entirely concentrated on a few majority classes.

## 3. Environment and Dependencies
To run this project, ensure you have the following environment set up:
* **OS:** Linux (Ubuntu) / Windows 10+
* **Python Version:** 3.8+
* **Hardware:** GPU recommended (e.g., NVIDIA RTX series / Google Colab)

Install the required dependencies using `pip`:
```bash
pip install torch torchvision datasets numpy matplotlib pandas
4. Training Instructions
To train the EfficientNet-V2-S model with Class-Weighted Loss and Custom Data Augmentation from scratch, run the train.py script. The script automatically downloads the Hugging Face dataset.

Bash
python train.py --batch_size 32 --epochs 20 --lr 0.001
Output: The script will save the best-performing model weights as best_model.pth in the current directory.

5. Evaluation Instructions
To evaluate the trained model on the test dataset and generate visual performance metrics, run the following command:

Bash
python evaluate.py --model_path ./best_model.pth
Output: This script will output the overall accuracy and generate evaluation graphs, including:

Qualitative results for minority classes (correct_predictions.jpg).

Trade-off analysis for majority classes (majority_compare.png).

A full 141x141 Confusion Matrix to visualize the prediction distribution.

6. How to Run the Inference Demo
We provide an inference script to test the model on individual images. You can pass a sample image to see the model's prediction and confidence score.

Bash
python inference_demo.py --model_path ./best_model.pth --image_path ./sample_image.jpg
Output: The script will display the input image and print the top-1 predicted fashion category along with its confidence percentage.

References:

Tan, M., & Le, Q. (2021). Efficientnetv2: Smaller models and faster training.

Cui, Y., et al. (2019). Class-balanced loss based on effective number of samples.

Install the required dependencies using the following command:
```bash
pip install torch torchvision datasets numpy matplotlib pandas
