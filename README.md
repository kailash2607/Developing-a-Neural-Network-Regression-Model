# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
The purpose of this experiment is to develop a Neural Network Regression Model using PyTorch to predict an output value based on a given input value. The problem is to train a neural network using the input-output data available in a CSV file so that it can learn the underlying relationship between the variables and predict outputs for new inputs.

Initially, the dataset is loaded using Pandas and divided into training and testing datasets using the train_test_split() function. Min-Max scaling is applied to normalize the input data. The training dataset is used to train the model, while the testing dataset is used to evaluate its ability to predict unseen data.

A feedforward neural network is constructed with one input neuron, two hidden layers containing 20 and 10 neurons, respectively, and one output neuron. ReLU activation functions are used in the hidden layers to learn nonlinear relationships. The model is trained for 1000 epochs using the Adam optimizer and Mean Squared Error (MSE) loss function to minimize the difference between actual and predicted outputs.

Finally, the training loss is plotted against the epochs to observe the learning process. The model is evaluated using the test dataset, and a prediction is generated for the input value 18. This experiment demonstrates how a neural network can learn from existing data, evaluate its predictive performance, and estimate output values for new inputs.


## Neural Network Model
<img width="922" height="802" alt="Screenshot 2026-10-23 100010" src="https://github.com/user-attachments/assets/7566d948-747f-4750-8f58-1a4314ee4b7b" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: KAILASH PRABHU S

### Register Number: 212224240068

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
```
```python
df = pd.read_csv("/content/Exp-1.csv")
df
x = df[["Input"]].values
y = df[["Output"]].values
```
```python
xt,xst,yt,yst=train_test_split(x,y,test_size=0.22,random_state=22)
```
```python
scaler1=MinMaxScaler()
scaler2=MinMaxScaler()
xt=scaler1.fit_transform(xt)
xst=scaler2.fit_transform(xst)
```
```python
xt1=torch.FloatTensor(xt)
xst1=torch.FloatTensor(xst)
yt1=torch.FloatTensor(yt)
yst1=torch.FloatTensor(yst)
```
```python
class neuralnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(
          nn.Linear(1,20),
          nn.ReLU(),
          nn.Linear(20,10),
          nn.ReLU(),
          nn.Linear(10,1)
        )
    def forward(self,x):
        return self.network(x)
```
```python
model=neuralnet()
criterion=nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr=0.01)
```
```python
epochs=1000
loss_list = []
for i in range(epochs):
    optimizer.zero_grad()
    pred=model(xt1)
    loss=criterion(pred,yt1)
    loss.backward()
    optimizer.step()
    loss_list.append(loss.item())
    if i%100==0:
        print(f"Epochs:{i} Loss:",loss.item())
```
```python
plt.plot(range(epochs), loss_list)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss vs Epochs")
plt.grid(True)
plt.show()
```
```python
with torch.no_grad():
    pred=model(xst1)
    loss=criterion(pred,yst1)
    print(loss.item())
```
```python
input=torch.FloatTensor(scaler1.transform([[18]]))
pred=model(input)
print(pred.item())
```

### Dataset Information
<img width="183" height="477" alt="Screenshot 2026-10-23 100655" src="https://github.com/user-attachments/assets/7dfbbff3-09be-4314-a1ba-4f149308d625" />


### OUTPUT

### Training Loss Vs Iteration Plot
<img width="642" height="500" alt="Screenshot 2026-10-23 100731" src="https://github.com/user-attachments/assets/206e1eaa-a294-4e3c-a924-aa8067397c76" />


### New Sample Data Prediction
<img width="493" height="127" alt="Screenshot 2026-10-23 100758" src="https://github.com/user-attachments/assets/2293fad7-eb9d-413a-b196-bbe1eeb36f8a" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
