#==================== IMPORT REQUIRED LIBRARIES =============================

import pandas as pd
from tkinter.filedialog import askopenfilename
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU
from keras.optimizers import SGD
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split



#========================= DATA SELECTION ==================================

filename = askopenfilename()
dataframe=pd.read_csv(filename)
print("==========================================")
print("-------------- Input Data ----------------")
print("==========================================")
print()
print(dataframe.head(20))

#============================= PREPROCESSING ==============================

#==== DROP UNWANTED COLUMNS ====

dataframe=dataframe.drop('datetime',axis=1)

#==== CHECKING MISSING VALUES ====

print("====================================================")
print("--------- Before Checking Missing Values ----------")
print("===================================================")
print()
print(dataframe.isnull().sum())

print("====================================================")
print("--------- After Checking Missing Values ----------")
print("===================================================")
print()
dataframe=dataframe.fillna(0)
print(dataframe.isnull().sum())


#=========================== DATA SPLITTING =======================

#=== TEST AND TRAIN ===

x=dataframe.drop('Global_active_power',axis=1)
y=dataframe.Global_active_power

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)

print("===============================================")
print("-------------- Data Splitting  ----------------")
print("===============================================")
print()

print("Total Data size is        :",dataframe.shape[0])
print("Total test Data size is   :",x_test.shape[0])
print("Total train Data size is  :",x_train.shape[0])


#=========================== MIN MAX SCALAR =======================

scaler = MinMaxScaler()

scaler.fit(x_train)

data_scaler=scaler.transform(x_test)

#============================= CLASSIFICATION ==============================

#=== LSTM ===

X=np.expand_dims(x_train, axis=2)
Y=np.expand_dims(y_train,axis=1)
 
#=== lstm architecture ===
model = Sequential()

#=== lstm layers ===
model.add(LSTM(input_shape=(x_train.shape[1],1), kernel_initializer="uniform", return_sequences=True, stateful=False, units=50))
model.add(Dropout(0.2))

model.add(LSTM(5, kernel_initializer="uniform", activation='relu',return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(3,kernel_initializer="uniform",activation='relu'))
model.add(Dense(1, activation='linear'))

#=== compile the model ===
model.compile(loss="mae", optimizer='adam',metrics=['mae','mse'])
model.summary()
print()

print("==================================================")
print("--------- Long Short Term Memory  ----------------")
print("==================================================")
print() 


#=== fitting the model ===
His=model.fit(X[0:10000],Y[0:10000], epochs = 5, batch_size=2000, verbose = 2)

Xx=np.expand_dims(x_test, axis=2)
Yy=np.expand_dims(y_test, axis=1)



#=== evaluate the model ===
mae_lstm1 =model.evaluate(X[0:10000],Y[0:10000], verbose=2)[1]

#=== predict the model ===
pred_lstm = model.predict(Xx[0:10000])

print("--------- Performance Analysis for LSTM  ----------------")
print()

from sklearn import metrics
mae_lstm = metrics.mean_absolute_error(pred_lstm,Yy[0:10000])
print("1.Mean Absolute Error :",mae_lstm1)
print()
mse_lstm = metrics.mean_squared_error(pred_lstm,Yy[0:10000])
print("2.Mean Squared Error :",mse_lstm)
print()
from math import sqrt

rmse_lstm=sqrt(mse_lstm)
print("3.Root Mean Squared Error :",rmse_lstm)

# mape = metrics.mean_absolute_percentage_error(pred_lstm,Yy[0:10000])

# from sklearn.metrics import mean_absolute_percentage_error

#==== GRU =====

# The GRU architecture
regressorGRU = Sequential()
# First GRU layer with Dropout regularisation
regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(x_train.shape[1],1), activation='tanh'))
regressorGRU.add(Dropout(0.2))
# Second GRU layer
regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(x_train.shape[1],1), activation='tanh'))
regressorGRU.add(Dropout(0.2))
# Third GRU layer
regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(x_train.shape[1],1), activation='tanh'))
regressorGRU.add(Dropout(0.2))
# Fourth GRU layer
regressorGRU.add(GRU(units=50, activation='tanh'))
regressorGRU.add(Dropout(0.2))
# The output layer
regressorGRU.add(Dense(units=1))
# Compiling the RNN
# regressorGRU.compile(optimizer=SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=False),loss='mae',metrics=['mae','mse'])

print("==================================================")
print("-------------------- GRU -------------------------")
print("==================================================")
print()
 

# Fitting to the training set
model.fit(X[0:10000],Y[0:10000],epochs=10,batch_size=150, verbose=2)


pred_gru = model.predict(Xx[0:10000])


mae_lstm =model.evaluate(X[0:10000],Y[0:10000], verbose=2)[1]



print("--------- Performance Analysis for GRU  ----------------")
print()

from sklearn import metrics
mae_gru = metrics.mean_absolute_error(pred_gru,Yy[0:10000])
print("1.Mean Absolute Error :",mae_gru)
print()
mse_gru = metrics.mean_squared_error(pred_gru,Yy[0:10000])
print("2.Mean Squared Error :",mse_gru)
print()
from math import sqrt

rmse_gru=sqrt(mse_gru)
print("3.Root Mean Squared Error :",rmse_gru)



#============================= COMPARISION ==============================


if mae_lstm1<mae_gru:
    print("=========================")
    print("LSTM is efficeint")
    print("=========================")
else:
    print("=========================")
    print("GRU is efficeint")
    print("=========================")
    


# Prediction Graph
import matplotlib.pyplot as plt
plt.plot(pred_gru[0:100])
plt.title("Energy Consumption")
plt.show()





























