# CIFAR-10 Image classification with Pytorch

## What is the objective?
The first main objective is write a code of validation. So I need to divide dataset and then write a code of train and valid in a one python file.
And the second main objective is comparing different ML models. So make models file directory to organize models.

## CIFAR input shape
input shape : (3 x 32 x 32)
The number 3 means RGB color channel. 

output shape : (10, 1) --> 10 classes

## CNN
convolution 2D
Maxpool 2D
These two filter has same attribute that they has a square filter. But convolution 2D has so many parameters but Maxpool just pick max value in its square filter. And generally maxpool2D squeeze output shape radically more than Conv2D.