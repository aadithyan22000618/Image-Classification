# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

The goal of this project is to develop a Convolutional Neural Network (CNN) for image classification using the Fashion-MNIST dataset. The Fashion-MNIST dataset contains images of various clothing items (T-shirts, trousers, dresses, shoes, etc.), and the model aims to classify them correctly. The challenge is to achieve high accuracy while maintaining efficiency.

## Neural Network Model

![image](https://github.com/user-attachments/assets/2254dcdc-73bc-4bd4-8567-fe7bdc9e91bd)

## DESIGN STEPS

STEP 1: Problem Statement

Define the objective of classifying handwritten digits (0-9) using a Convolutional Neural Network (CNN).

STEP 2:Dataset Collection

Use the MNIST dataset, which contains 60,000 training images and 10,000 test images of handwritten digits.

STEP 3: Data Preprocessing

Convert images to tensors, normalize pixel values, and create DataLoaders for batch processing.

STEP 4:Model Architecture

Design a CNN with convolutional layers, activation functions, pooling layers, and fully connected layers.

STEP 5:Model Training

Train the model using a suitable loss function (CrossEntropyLoss) and optimizer (Adam) for multiple epochs.

STEP 6:Model Evaluation

Test the model on unseen data, compute accuracy, and analyze results using a confusion matrix and classification report.

STEP 7: Model Deployment & Visualization

Save the trained model, visualize predictions, and integrate it into an application if needed.

## PROGRAM

### Name:AADITHYAN R 
### Register Number:212222230001
```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
    self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.fc1 = nn.Linear(128*3*3,128)
    self.fc2 = nn.Linear(128,64)
    self.fc3 = nn.Linear(64,10)

  def forward(self,x):
    x = self.pool(torch.relu(self.conv1(x)))
    x = self.pool(torch.relu(self.conv2(x)))
    x = self.pool(torch.relu(self.conv3(x)))
    x = x.view(x.size(0),-1)
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = self.fc3(x)
    return x

```

```python
# Initialize the Model, Loss Function, and Optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

```

```python
# Train the Model
def train_model(model, train_loader, num_epochs=3):

     for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images,labels in train_loader:
      optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs,labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()

    print('Name: AADITHYAN R')
    print('Register Number: 212222230001')
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

        
        
        
        print('Name: AADITHYAN R ')
        print('Register Number:212222230001 ')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

```

## OUTPUT
### Training Loss per Epoch

<img width="240" height="155" alt="Screenshot (9)" src="https://github.com/user-attachments/assets/f738caab-be75-4fd3-bfb7-b8cc1f4ead66" />


### Confusion Matrix


<img width="619" height="451" alt="Screenshot (7)" src="https://github.com/user-attachments/assets/d38d4b3d-d6ac-4e92-b034-1ef3dce41b8a" />

### Classification Report

<img width="393" height="283" alt="Screenshot (8)" src="https://github.com/user-attachments/assets/52fb496e-f8f7-48b4-9298-1d02bce602f2" />



### New Sample Data Prediction

<img width="337" height="373" alt="Screenshot (6)" src="https://github.com/user-attachments/assets/6a88a596-89c2-4f8c-be71-443207f6f006" />


## RESULT
Thus, We have developed a convolutional deep neural network for image classification to verify the response for new images.
