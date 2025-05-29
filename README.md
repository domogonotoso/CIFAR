# CIFAR-10 Image classification with Pytorch

## What is the Goals of This Project?
1. Implement a validation split within the training script to evaluate generalization performance.

2. Compare different machine learning models by organizing them into a dedicated models/ directory.

3. Design a modular project structure for clarity, maintainability, and scalability.

````markdown
📁 The structure of this CIFAR-10 classification project:
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
 

## ✅ Implemented Features

- Structured project into modular directories (models/, utils/, results/, etc.)
- Compared two CNN architectures: a modified MnistCNN and ResNet-18
- Created training and validation accuracy plots for performance monitoring
- Implemented training-validation split within the training script
- Applied early stopping to prevent overfitting based on validation accuracy
- Automatically saved both the best-performing and last models
- Enabled easy switching between model architectures (`MnistCNN` ↔ `ResNet18`) in scripts


## Dataset Information

- **Input shape**: (3 × 32 × 32) — RGB images (height × width = 32 × 32)
- **Number of classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)


## CNN Components

Both **Conv2D** and **MaxPool2D** operations use square kernels by default (e.g., 3×3). However, their purposes and characteristics differ:

- **Conv2D**: Applies learnable filters to extract features from the input. Each filter has weights that are updated during training, allowing the network to learn spatial patterns.
- **MaxPool2D**: Performs downsampling by selecting the maximum value in each kernel region. It has no learnable parameters and is typically used to reduce spatial dimensions and computation.

Note: MaxPooling usually reduces the feature map size more aggressively than convolution alone.


## BatchNorm2d
BatchNorm2d normalizes each feature map across the batch, then applies a learnable scale (γ) and shift (β) to retain representation capacity.

## Two Models

The first model is adapted from the one used in MNIST classification to work with CIFAR-10.  
The second is ResNet-18, which features skip connections as a key concept.

## ResNet

### What is a skip connection?

Skip connections help preserve original features by directly adding the input tensor to the output of later layers. This facilitates gradient flow in deep networks and prevents information loss.

```python
def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    out += self.shortcut(x)  # BasicBlock inherently includes a skip connection.
    return F.relu(out)
```

## Runtime Error While Running train.py

While running the training script, the following error occurred:

```text
File "C:\Users\wlskr\Downloads\ChanKyu_Kim\CIFAR\models\MNIST.py", line 21, in forward
    x = F.relu(self.fc1(x))
RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x4096 and 3136x128)
```

    To fix this, I modified the fully connected layer in MNIST.py:
    from
    ```code
    self.fc1 = nn.Linear(64 * 7 * 7, 128)
    ```
    to
    ```code
    self.fc1 = nn.Linear(64 * 8 * 8, 128)
    ```

Since I'm familiar with linear transformations and tensor shapes, I was able to quickly identify that the mismatch came from the input size change — CIFAR-10 uses 32×32 images instead of 28×28 like MNIST.


Training with the adjusted model and testing various hyperparameters:
[Epoch 10] Train Loss: 0.6118
[Epoch 10] Validation Accuracy: 73.09%
Model saved to model.pt

At first, it seemed like the model was underfitting, so I considered increasing the number of epochs.
However, after running it for 20 epochs, I observed signs of overfitting — validation accuracy plateaued while training loss continued to decrease.
As a result, I decided to stick with 10 epochs and apply early stopping.




## Training Progress

### MnistCNN

![Training Accuracy vs Validation Accuracy](results/Mnist_t&v.png)

> **Note:** The training accuracy starts lower than the validation accuracy because it is measured before any learning occurs, whereas validation accuracy is evaluated after each epoch.

- Early stopping at epoch 14  
- 🏆 Best model saved to `model_best.pth` (Validation Accuracy: **73.63%**)  
- 📦 Last model saved to `model.pth` (Validation Accuracy: **72.56%**)  
- ✅ Best model Test Accuracy: **72.44%**  
- ✅ Last model Test Accuracy: **72.38%**

---

### ResNet-18

![Training Accuracy vs Validation Accuracy](results/ResNet_t&v.png)

- Early stopping at epoch 15  
- 🏆 Best model saved to `model_best.pth` (Validation Accuracy: **82.83%**)  
- 📦 Last model saved to `model.pth` (Validation Accuracy: **81.66%**)  
- ✅ Best model Test Accuracy: **82.90%**  
- ✅ Last model Test Accuracy: **81.99%**


🧠 **Summary:** ResNet-18 outperformed MnistCNN by ~10% in validation and test accuracy, highlighting the effectiveness of deeper architectures on complex datasets like CIFAR-10.

---

### Switching Between Models

To switch from ResNet-18 to the MnistCNN, update the model class in both `train.py` and `test.py`:

```python
# Change this:
ResNet18()

# To this:
MnistCNN()


```

