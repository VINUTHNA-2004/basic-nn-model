# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

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

## PROGRAM

```
## Importing modules

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

## Authenticate & Create data frame using data in sheets

from google.colab import auth
import gspread
from google.auth import default
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('exp1').sheet1
data = worksheet.get_all_values()
dataset1=pd.DataFrame(data[1:],columns=data[0])
dataset1=dataset1.astype({'Input':'float'})
dataset1=dataset1.astype({'Output':'float'})
dataset1.head()

## Assign X & Y Values

X = dataset1[['Input']].values
y = dataset1[['Output']].values
X

## Normalize the values and split the data

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train) 
## Create a neural network and train it.
ai_brain=Sequential([
    Dense(8,activation='relu'),
    Dense(10,activation='relu'),
    Dense(1)
])
ai_brain.compile(optimizer='rmsprop',loss='mse')
ai_brain.fit(X_train1,y_train,epochs=200)

## Plot the loss

loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()

## Predict for some value

X_test1 = Scaler.transform(X_test)
X_n1 = [[30]]
X_n1_1 = Scaler.transform(X_n1)
ai_brain.predict(X_n1_1)

```

## Dataset Information
![ex ](https://github.com/VINUTHNA-2004/basic-nn-model/assets/95067307/7b35d6b3-b6b5-4702-8417-54b2e4c64343)


## OUTPUT

### Training Loss Vs Iteration Plot
![ex1](https://github.com/VINUTHNA-2004/basic-nn-model/assets/95067307/51e69105-c805-49b7-8ba6-77e2a68f0a12)



### Test Data Root Mean Squared Error

![ex1 2](https://github.com/VINUTHNA-2004/basic-nn-model/assets/95067307/f6388d36-08fb-4f5f-b4c0-3555e7169727)

### New Sample Data Prediction
![ex1 1](https://github.com/VINUTHNA-2004/basic-nn-model/assets/95067307/56689ddb-31eb-456f-98be-24fc3025fffe)


## RESULT:
A Basic neural network regression model for the given dataset is developed successfully.
