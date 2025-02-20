# StyleGAN2-ADA for Crack Image Generation

This repository contains a StyleGAN2-ADA model fine-tuned to generate synthetic crack images of class 4.  
This project is part of MIIRAG-Task-2 and aims to improve crack detection datasets by augmenting real data with synthetic cracks.

---

## Based On

This project is based on two repositories:

- Pretrained Textures Model from [justinpinkney/awesome-pretrained-stylegan2](https://github.com/justinpinkney/awesome-pretrained-stylegan2) (Used as the initial model).
- Training Code from [NVlabs' StyleGAN2-ADA-PyTorch](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/train.py) (Used for fine-tuning).

Modifications were made to adapt these models for crack image generation.

---

    ## Key Modifications

    ### Model
    - Fine-tuned a StyleGAN2 model pretrained on textures to generate synthetic crack images.
    - Adjusted augmentation settings to preserve crack structures without unnecessary distortions.

    ### Compatibility Updates
    - Updated the outdated StyleGAN2-ADA repository for compatibility with modern PyTorch versions.

---

## Dataset

- The model was trained on class 4 of the EDM600 crack image dataset, which consists of crack images divided into 8 classes (`class_0` to `class_7`).
- Images were preprocessed and stored as TFRecords for efficient training.

---

## Model Output & Checkpoints

- Training snapshots are saved every 5 ticks (`--snap=5`).
- Final trained model weights are available for inference and further fine-tuning.

---

## How to Use This Repository

### 1️ - Prepare Your Dataset

Convert images into TFRecords for efficient training using :  
python dataset_tool.py --source=/path/to/images --dest=/path/to/tfrecords --resolution=512x512
(The dataset_tool.py script can be found in the StyleGAN2-ADA-PyTorch folder.)

### 2 - Train the Model

fine-tune the model with optimized augmentation settings:
!python train.py --resume=texture.pkl --outdir=pathtotheoutputfolder \
--data=pathToDataset --gpus=2 --cfg=paper512 --snap=5 --metrics=none \
--augpipe=cf --mirror=1 --batch=32 --kimg=15000
(The train.py script can be found in the StyleGAN2-ADA-PyTorch folder.)
(The notebook StyleGAN2_model_training_notebook.ipynb includes the full training pipeline.)

### 3 - Use the Trained Model

generate images using the trained model checkpoint:
python generate.py --network=model.pkl --seeds=1,2,3 --trunc=1 --outdir=output/
Model Weights Available in This Repository
(The generate.py script can be found in the StyleGAN2-ADA-PyTorch folder.)

## Model Weights Available in This Repository

texture.pkl → Pretrained model (textures-based, initial model).
6network-snapshot-000220.pkl → Fine-tuned model (trained on crack images, latest checkpoint).
These weights can be used for further fine-tuning
