# CIFAR-10 Image classification with Pytorch

## What is the objective?
1. Implement validation logic by splitting the dataset into training and validation sets within a single Python file.

2. Compare different machine learning models by organizing them into a dedicated models/ directory.

3. Build a clear and modular project structure for maintainability and scalability.

````markdown
📁 The structure of CIFAR project
````

```text
CIFAR/
├── data/              # Raw dataset files (e.g., CIFAR-10)
├── models/            # CNN model definitions (MnistCNN, ResNet-18)
├── results/           # Training and validation plots
├── utils/             # Utility functions (data loading, plotting, etc.)
├── train.py           # Training pipeline
├── test.py            # Evaluation script
├── requirements.txt
├── .gitignore
└── README.md
```
 


## Dataset Information

- **Input shape**: (3 × 32 × 32), RGB images
- **Number of classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)


## CNN Components

Both **Conv2D** and **MaxPool2D** operations use square kernels by default (e.g., 3×3). However, their purposes and characteristics differ:

- **Conv2D**: Applies learnable filters to extract features from the input. Each filter has weights that are updated during training, allowing the network to learn spatial patterns.
- **MaxPool2D**: Performs downsampling by selecting the maximum value in each kernel region. It has no learnable parameters and is typically used to reduce spatial dimensions and computation.

Note: MaxPooling usually reduces the feature map size more aggressively than convolution alone.


## BatchNorm2d
After batch normalization, elements of batch are multiplied by r(gamma) and added by b(beta). They are parameters batch normarlization.

## Two model
The first model is what we used before at MNIST classification. Just revise slightly to fit changed datasets CIFAR-10.
And the second model is Resnet-18. Skip connection is important concept of it.

## Resnet

What is skip connection?
Not to forgot original information, add original tensor to output tensor.
```python
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Basicblock inheritly has skip connection.
        return F.relu(out)   
```

## Runtime error while running train.py
  File "C:\Users\wlskr\Downloads\ChanKyu_Kim-20250522T014702Z-1-001\ChanKyu_Kim\CIFAR\models\MNIST.py", line 21, in forward
    x = F.relu(self.fc1(x))
    RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x4096 and 3136x128)

    So revise code at MNIST.py
    from
    ```code
    self.fc1 = nn.Linear(64 * 7 * 7, 128)
    ```
    to
    ```code
    self.fc1 = nn.Linear(64 * 8 * 8, 128)
    ```

    Since I'm more familiar with theoretical concepts, I can quickly spot where things need to be fixed when it comes to matrices or linear transformations.


    -Trying with hyperparameters.
    [Epoch 10] Train Loss: 0.6118
    [Epoch 10] Validation Accuracy: 73.09%
    Model saved to model.pt

    Model seems to be underfitting. It need higher epochs.
    >>> No, I can see that after 10 epochs, train loss sustainly decreases but accuracy just stay around 70%. With 20 epochs, model seems to be overfitted. So just maintain 10 epochs and use early stoping.


## What to do?
Add train loss for checking overfitting --O  
test.py --O  
Graph --O  
early stopping --O   
 save best model --O  
Understanding the code nad strucutre of Resnet-18   
Drawing structure of file directories of the project.


## Training Progress 

  ![Training Accuracy vs Validation Accuracy](results/Mnist_t&v.png)

  > **Note:** Train accuracy starts lower than validation accuracy because it is measured before the model has learned anything, while validation is evaluated after the first epoch.

  ### MnistCNN
  Early stopping at epoch 14
  Best model saved to model_best.pth (Val Acc: 73.63%)
  Last model saved to model.pth (Val Acc: 72.56%)
  ✅ Best model Test Accuracy: 72.44%
  ✅ Last model Test Accuracy: 72.38%

  ### Resnet18
  ![Training Accuracy vs Validation Accuracy](results/ResNet_t&v.png)

  Early stopping at epoch 15
  Best model saved to model_best.pth (Val Acc: 82.83%)
  Last model saved to model.pth (Val Acc: 81.66%)
  ✅ Best model Test Accuracy: 82.90%
  ✅ Last model Test Accuracy: 81.99%




If you want to do with MNIST.py, just change ResNet18 to MnistCNN at train.py and test.py.
dropout to conv feature map... But I have less computing resource. So 