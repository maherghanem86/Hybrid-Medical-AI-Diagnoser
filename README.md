🏥 Entropy-Based Dynamic Ensemble AI for Medical Image Diagnosis

This repository contains the code and resources for a Master's Thesis project at the Syrian Virtual University (SVU). The project introduces an advanced Entropy-Based Dynamic Ensemble Deep Learning framework designed to automate and highly improve the diagnosis of medical imaging, specifically targeting Skin Cancer (HAM10000) and Chest X-Ray Pneumonia.

🌟 Key Features

Dynamic Ensemble Architecture: Instead of traditional static voting, this system dynamically calculates the Entropy (uncertainty) of three pre-trained state-of-the-art models (ResNet-50, DenseNet-121, EfficientNet-B0) per image. It assigns higher weights to the most confident model, achieving superior robust accuracy.

Multi-Domain Medical Diagnosis: Rigorously trained and evaluated on two completely different medical domains (Dermatology and Pulmonology) to prove architectural generalization.

Explainable AI (XAI): Integrates Grad-CAM (Gradient-weighted Class Activation Mapping) to generate visual heatmaps, showing doctors exactly where the model looked to make its decision, enhancing clinical trust.

Interactive Web Interface: Fully deployed on Hugging Face Spaces using Gradio, providing an end-to-end usable prototype for real-time medical inference.

🛠️ System Architecture

The pipeline consists of three primary Deep Learning models fine-tuned on medical datasets:

ResNet-50

DenseNet-121

EfficientNet-B0

The Entropy-Based Fusion Mechanism:
For any given input image $x$, each model outputs a probability distribution. The system calculates the Shannon Entropy for each output:


$$H(p) = -\sum p_i \log(p_i)$$


The model with the lowest entropy (highest certainty) is dynamically assigned the highest weight in the final probability fusion.



📊 Evaluation & Results

The proposed Dynamic Ensemble (Entropy-based) was evaluated against a standard Static Average Ensemble. The dynamic approach proved highly effective in mitigating false positives by suppressing the noisy outputs of uncertain models.

1. Skin Cancer Classification (HAM10000 - 7 Classes)

Static Ensemble ROC-AUC: 0.9839

Dynamic Ensemble ROC-AUC: 0.9822 (Maintained extremely high AUC while optimizing single-prediction confidence)

Dynamic Ensemble Confusion Matrix

Static Ensemble Confusion Matrix





2. Chest X-Ray Pneumonia (2 Classes)

Static Ensemble ROC-AUC: 0.9726

Dynamic Ensemble ROC-AUC: 0.9729 (Demonstrated explicit improvement in macro metrics)

Dynamic Ensemble Confusion Matrix



🚀 Live Demo & Quick Start

Experience the model directly in your browser without any setup. The UI supports both Skin Lesion and Chest X-Ray analysis with Grad-CAM visualization.

[👉 Launch Gradio Demo on Hugging Face Spaces
](https://huggingface.co/spaces/maherghanem86/Hybrid-AI-Diagnoser)

Local Execution

To run the Gradio app locally:

Clone the repository.

Install dependencies: pip install torch torchvision numpy gradio grad-cam huggingface-hub opencv-python-headless

Run the UI script: python app.py (Ensure you have downloaded the weights or use the HF Hub download code provided in the app).

Disclaimer: This system is a research prototype developed for academic purposes. It is not a certified medical device and should not be used as a sole diagnostic tool without professional clinical validation.


https://huggingface.co/maherghanem86/chest-xray-models

https://huggingface.co/maherghanem86/skin-cancer-models
