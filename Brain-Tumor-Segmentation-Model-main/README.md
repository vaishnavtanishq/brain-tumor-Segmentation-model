# Brain-Tumor-Detection-Segmentation-Model

The model generated a segmentation map with a given input MRI scan image to classify a tumor region pixel-wise with
around 3900 training images.

Compared accuracy with FPN and U-Net model with different backbones.Replaced encoder part(backbone) of U-Net model
with pre-trained resnet-101 model using Transfer Learning and trained only decoder-part to achieve highest accuracy.

Applied most optimised focal-tversky loss function to get the control of non-linear loss behaviour and data imbalance which
results the tversky score of 92% and 90% dice-coefficient.

Created a desktop app using PySimpleGUI.
