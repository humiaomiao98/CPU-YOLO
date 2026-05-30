# CPU-YOLO: A Riverine Garbage Detection Method Based on YOLOv11n

 Important Notice

To facilitate the peer-review process, this repository currently provides only partial core code snippets, a limited set of basic sample data, and a demonstration framework for execution, with the aim of supporting the validity and existence of the method proposed in this paper.

**Upon formal acceptance and online publication of this paper, all source code, pre-trained model weights, and the complete dataset information will be fully released in this repository. We sincerely appreciate your attention and understanding！**

---

# 1.  Abstract

This paper proposes CPU-YOLO, an improved riverine garbage detection method based on YOLOv11n that incorporates Cascade Group Attention, a P2 small-object detection layer, and the UIoU loss function to enhance feature discrimination, small-object detection, and bounding box regression robustness in complex riverine environments. 

# 2.  Experimental Results

Experiments on the My-Flow dataset and two public datasets demonstrate that the proposed method achieves consistent improvements in precision, recall, and mAP while maintaining a real-time inference speed of 384.6 FPS.

#  Dataset

To ensure the comprehensiveness and reliability of the experimental results, an experimental dataset named My-Flow is constructed by combining original images with augmented images. The original images consist of 440 self-collected riverine garbage images, 350 images from the publicly available Flow-Img dataset, and 350 images from the IWHR_AI_Lable_Floater_V1 dataset released by the China Institute of Water Resources and Hydropower Research.

Representative samples of the dataset.

<img width="229" height="162" alt="image" src="https://github.com/user-attachments/assets/3485af45-a90a-44f0-88af-cebfae97c709" />
<img width="219" height="163" alt="image" src="https://github.com/user-attachments/assets/59ced4d7-c669-441e-b8b1-0a3d82c27830" />
<img width="225" height="162" alt="image" src="https://github.com/user-attachments/assets/bc42e50c-b8bd-417a-8e04-9a0f141ff374" />
<img width="213" height="161" alt="image" src="https://github.com/user-attachments/assets/b8c5ca9a-0257-40da-8aa9-8269af534f28" />

---

# 3. Repository Structure

The current directory structure of this repository and the corresponding locations of its contents are organized as follows:

```
CPU-YOLO
├── datasets/            # Sample dataset directory
├── models/              # Model architecture directory
├── tools/               # Directory for running and testing scripts
├── visualization/       # Directory for storing visualization result images.
├── requirements.txt     # Description of the environment dependency configuration file.
└── README.md            # This documentation file.
 Directory Details

To illustrate the fundamental operational workflow of the project, the repository includes the following representative sample components:

- datasets/: A directory for storing training and testing data. Currently, it contains only a very limited number of sample images, primarily intended to demonstrate the data input format and annotation protocol used in this study. The complete dataset will be made publicly available upon acceptance of the paper.
- models/: A directory for model-related code. At present, only partial implementations of basic module components are provided to demonstrate the existence of the proposed model architecture. The complete network configuration files will be released in a subsequent update.
- tools/: A directory for related execution scripts. Currently, a simplified demonstration file is provided to illustrate the basic operational workflow of inference or evaluation.
- requirements.txt: Lists the fundamental environment dependencies required for running the project.
   ```
# 4. Quick Start

## Environment Setup

We recommend creating a new Python environment and installing all required dependencies using:

```bash
pip install -r requirements.txt
```

## Experimental Configuration

The experiments reported in this repository were conducted under the following environment:

| Component        | Configuration                        |
| ---------------- | ------------------------------------ |
| Operating System | Windows 11                           |
| Python           | 3.9                                  |
| PyTorch          | 2.0.0                                |
| CUDA             | 12.9                                 |
| GPU              | NVIDIA GeForce RTX 5090 D v2 (24 GB) |
| CPU              | Intel Core i7-11700F                 |
| RAM              | 32 GB                                |

## Training Configuration

The CPU-YOLO model was trained using the following settings:

| Parameter      | Value     |
| -------------- | --------- |
| Input Size     | 640 × 640 |
| Epochs         | 600       |
| Batch Size     | 8         |
| Optimizer      | SGD       |
| AMP            | Enabled   |
| Workers        | 2         |
| Close Mosaic   | 30        |
| Early Stopping | Disabled  |

## Dataset Preparation

Please organize the dataset according to the following structure:

```text
datasets/
├── images
│   ├── train
│   ├── val
│   └── test
└── labels
    ├── train
    ├── val
    └── test
```

The dataset configuration file should be placed at:

```text
datasets/data.yaml
```

## Training

Launch model training with:

```bash
python train.py
```

The trained weights will be automatically saved to:

```text
runs/train/weights/
```

## Project Structure

```text
CPU-YOLO/
├── datasets/
├── ultralytics/
├── train.py
├── predict.py
├── benchmark.py
├── requirements.txt
├── README.md
└── CPU-YOLO.yaml
```

---
# 5. Citation

If you find our work helpful, we kindly encourage you to cite our paper after it has been formally accepted for publication:

```bibtex
@article{
  title={CPU-YOLO: A Riverine Garbage Detection Method Based on YOLOv11n},
 
  year={2026}
}
```
