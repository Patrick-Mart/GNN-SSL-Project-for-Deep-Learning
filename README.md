# GNN-SSL-Project-for-Deep-Learning

This repository contains the Self-Supervised Graph Neural Network in python project realised for the Deep Learning course at DTU.

## Necessary libraries

To run the different piece of code, it is necessary to have an environment with:
*torch (version <= 2.8.0)
*torch geometric for the corresponding version of pytorch
*matplotlib for plotting graph
*sklearn for PCA.

## Results

Considering the size of the dataset and the models we were working with, we used the HPC of DTU to train our models. A jupyter notebook containing all the results is thus not available for our project.
To view the results obtained, for the supervised prediction the file supervised.py need to be run. For training the encoder/decoder pair the file unsupervised_features.py. For training the classifier on top of the encoder the file self_supervised.py needs to be run.
All graph are displayed in the results folder.
