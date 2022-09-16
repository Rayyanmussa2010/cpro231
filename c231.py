import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

dataset=pd.read_csv(r'books.csv', error_bad_lines=False)
x=dataset.iloc[:,[4,10]].values
y=dataset.iloc[:,3].values
model=Sequential()
model.add(Dense(128, input_dim=8, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(56, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()