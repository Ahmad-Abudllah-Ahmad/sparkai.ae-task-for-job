# Technical Skills Assessment Filter - Project Analysis

This document provides a detailed mapping of the "AI / Machine Learning Engineer – Technical Skills Assessment" against the implemented EuroSAT Land Use Classification System. It serves to strictly answer the assessment queries and demonstrate the engineering decisions made during development.

---

## 1. Assessment Requirement Mapping

The following table maps strictly to the requirements outlined in the assessment document.

| Section | Requirement | Project Implementation & Decision Rationale |
| :--- | :--- | :--- |
| **Core Task** | **Problem Statement** | **Image Classification** selected. Using **EuroSAT** dataset (RGB version). <br> *Rationale:* EuroSAT is non-trivial (like CIFAR/MNIST) but manageable size-wise (~27k images), allowing for meaningful model comparison (CNN vs ResNet) within the 72h timeframe. It has clear real-world utility (environmental monitoring). |
| **1. Data** | **Understanding & Prep** | - **Size:** 27,000 images, 10 classes. <br> - **Format:** 64x64 RGB images. <br> - **Class Distribution:** Relatively balanced (~2k-3k per class). <br> - **Imbalance Handling:** Computed class weights (inverse frequency) and used `WeightedRandomSampler` in training to ensure the model doesn't bias towards majority classes like 'AnnualCrop' or 'Forest'. |
| | **Preprocessing** | **Training:** Random Rotation, Horizontal/Vertical Flips (satellite images are invariant to rotation/flip), Color Jitter. <br> **Inference:** Resize (to 64x64/224x224), Normalization (using ImageNet stats for ResNet), ToTensor. |
| **2. Modeling** | **Model Selection** | **Model 1: Baseline CNN** - A custom lightweight 3-layer CNN. Chosen to demonstrate ability to build models from scratch and serve as a speed/size benchmark. <br> **Model 2: ResNet18 (Fine-Tuned)** - Transfer learning from ImageNet. Justified because satellite imagery shares low-level visual features (textures/edges) with natural images, speeding up convergence and improving accuracy. |
| | **Compute Constraints** | **ResNet18** was chosen over ResNet50/101 specifically for **CPU/Laptop inference**. <br> - **Size:** 45MB vs ~100MB+ for larger models. <br> - **Inference:** ~65ms on CPU, making it feasible for real-time deployment without GPU. <br> - **Training:** Fine-tuning only the last 2 layers initially avoids heavy backprop costs. |
| **3. Evaluation** | **Methodology** | **Split:** Stratified 70% Train / 15% Val / 15% Test. Stratification ensures all land use classes are represented equally in evaluation. <br> **Metrics:** Accuracy, Precision, Recall, Macro F1-Score (crucial since classes are distinct yet related). |
| | **Error Analysis** | Implemented `plot_confusion_matrix` to visualize misclassifications. Example finding: "River" and "Highway" can differ subtly in low-res satellite imagery; the confusion matrix helps identify these specific crossover errors. |
| **4. Inference** | **Pipeline Design** | Clean separation: `api/index.py` handles HTTP/Validation, while `src/inference` manages model logic. <br> **Loading:** Lazy loading pattern used to prevent memory spikes on startup (serverless friendly). |
| | **Exposition** | **FastAPI** REST API. <br> - Endpoint: `POST /api/predict` <br> - Docs: Auto-generated Swagger/Redoc at `/docs`. <br> - Dockerized: Yes, `Dockerfile` provided for consistent serving. |
| **5. Engineering** | **Code Quality** | - **Structure:** Modular (`src/data`, `src/models`, `src/training`) rather than a monolithic script. <br> - **Config:** `configs/config.yaml` manages all hyperparameters (learning rate, batch size, paths) centrally. <br> - **Logging:** Python `logging` used instead of `print` statements for production readiness. |

---

## 2. In-Depth Analysis

### Data Understanding
The EuroSAT dataset presents a unique challenge compared to standard object photos. Being satellite imagery:
1.  **Orientation Invariance:** A forest looks like a forest whether viewed from North or South up. This justified using aggressive **RandomHorizontalFlip** and **RandomVerticalFlip** augmentations.
2.  **Resolution:** At 64x64, features are coarse. Upscaling to 224x224 for ResNet was necessary to utilize pretrained weights effectively, even though it adds compute overhead.

### Model Selection Strategy
-   **Baseline CNN:** Achieved ~95% accuracy but is extremely lightweight (1.3MB). This proves that for specific, constrained domains, custom architectures can be very competitive.
-   **ResNet18:** Achieved ~92% accuracy (slightly lower than baseline in this specific run, likely due to domain shift or hyperparameter tuning needs for the specific split). However, its strength lies in **robustness** and generalization potential if more data were added. It was chosen to demonstrate **Transfer Learning** competence.

### Engineering Decisions
-   **Docker:** Used `python:3.11-slim` to keep image size small.
-   **API:** Separated the prediction logic (`predict` function) from the route handler. This allows the inference logic to be used by a script or a different API framework without rewriting code.

---

## 3. Future Improvements (Scale & Compute)

**If additional compute and time were available, how would we scale?**

### 1. Scaling Training
*   **Distributed Data Parallel (DDP):** Use `torch.nn.parallel.DistributedDataParallel` to train across multiple GPUs.
*   **Larger Batch Size:** Increase batch size from 64 using gradient accumulation to stabilize batch norm statistics.
*   **Mixed Precision:** Implement `torch.cuda.amp` (FP16) training to reduce VRAM usage and speed up training on Tensor Core GPUs.

### 2. Model Improvements
*   **Vision Transformers (ViT):** Experiment with ViT-Tiny or Swin Transformers, which often capture global context better in remote sensing data than CNNs.
*   **Self-Supervised Learning:** Pre-train on a massive unlabeled satellite dataset (like Sentinel-2 archives) using methods like **SimCLR** or **DINO** before fine-tuning on EuroSAT. This handles the "labeled data scarcity" problem common in ML.

### 3. Data Improvements
*   **Test Time Augmentation (TTA):** During inference, predict on the image + its flipped/rotated versions and average the results to improve confidence.
*   **Hard Example Mining:** Automatically identify samples with high loss (e.g., River vs Highway) and oversample them in subsequent training epochs.
