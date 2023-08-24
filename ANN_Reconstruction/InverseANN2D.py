import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
import re
import csv
from keras import backend as K

# Number of training points
NUM_TP=1000000

# Feed file content
Feed=np.zeros([NUM_TP,6])

# File for the number of counts
filename_counts = 'interpolaited_counts_all_datasetC.txt'
data_counts = np.loadtxt(filename_counts, delimiter='\t')

for i in range(4):
    Feed[:,i] = data_counts[0:NUM_TP, i]

# File for the position x and y
filename_pos = 'x_y_robot_position_datasetC.txt'
data_pos = np.loadtxt(filename_pos, delimiter=',')


Feed[:,4] = (data_pos[0:NUM_TP, 0]-670.006)/1000
Feed[:,5] = (data_pos[0:NUM_TP, 1]+199.486)/1000

# Write the counts and position on a text file
with open("Feed.txt", "w") as file_output: 
    file_output.truncate(0)
    for row in Feed:
        line = '\t'.join([str(item) for item in row])
        file_output.write(f'{line}\n')
       
#Define input features 
X= keras.Input(shape=(1,), name="X")
Y= keras.Input(shape=(1,), name="Y")


# Joining strings together end-to-end to create a new string to feed the model
# The new string is input layer
x = layers.concatenate([X,Y])


#Define the hidden layers
hidden1 = layers.Dense(64, activation='tanh')(x)
hidden2 = layers.Dense(32, activation='tanh')(hidden1)
hidden3 = layers.Dense(16, activation='tanh')(hidden2)

#Define the output layers
Count1 = layers.Dense(1,activation='linear', name="Count1")(hidden3)
Count2 = layers.Dense(1,activation='linear', name="Count2")(hidden3)
Count3 = layers.Dense(1,activation='linear', name="Count3")(hidden3)
Count4 = layers.Dense(1,activation='linear', name="Count4")(hidden3)

model = keras.Model(inputs=[X,Y],outputs=[Count1,Count2,Count3,Count4],)


keras.backend.set_epsilon(1)
model.compile(
    optimizer='Adam',
    loss=['mse','mse','mse','mse'],
    loss_weights=[1.0, 1.0,1.0,1.0],
    metrics=['MAE','mean_absolute_percentage_error']   
)

# Define the learning rate
LR=0.001
K.set_value(model.optimizer.learning_rate,LR)

pd_dat = pd.read_csv('Feed.txt', delimiter='\t')

# Extract the values from the dataframe
dataset = pd_dat.values

#Normalizing the input counts
X_raw=dataset[:,4:]
scaler_X = MinMaxScaler()
scaler_X.fit(X_raw)
X_scale= scaler_X.transform(X_raw)

# Training and testing the Neural Network
X_train, X_test, Y_train, Y_test = train_test_split(X_scale,dataset[:,:4], test_size=0.2)

x_train, y_train = np.transpose(X_train)
x_test, y_test = np.transpose(X_test)

with open('test.txt','w') as file:
	for i in range(len(Y_train)):
		file.write(str(Y_train[i,:])+'\n')

count1_train, count2_train, count3_train, count4_train = Y_train[:,0], Y_train[:,1], Y_train[:,2], Y_train[:,3]
count1_test, count2_test, count3_test, count4_test = Y_test[:,0], Y_test[:,1], Y_test[:,2], Y_test[:,3]


inputs_train=[x_train, y_train]
outputs_train=[count1_train, count2_train, count3_train, count4_train]


history=model.fit(inputs_train,outputs_train,
                 validation_split=0.3,
                 epochs=8000,
                 batch_size=50000,
                 )
print(history.history)


result=model.evaluate([x_test,y_test],[count1_test, count2_test, count3_test, count4_test],verbose=2)
print(result)


pyplot.subplot(211)
pyplot.title('Loss RUN 4')
plt.semilogy(history.history['loss'], label='train')
plt.semilogy(history.history['val_loss'], label='validation')
plt.xscale("log")
pyplot.legend()

# In[2]
Start=100000
Stop=250000
Step=100

Pred = np.zeros([len(data_pos[Start:Stop:Step,0]),2])
Pred[:,0]=(data_pos[Start:Stop:Step, 0]-670.006)/1000
Pred[:,1]=(data_pos[Start:Stop:Step, 1]+199.486)/1000

X_pre=Pred
scaler_X_pre = MinMaxScaler()
scaler_X_pre.fit(X_pre)
X_scale_pre= scaler_X.transform(X_pre)
x_pre, y_pre=np.transpose(X_scale_pre)
prediction=model.predict([x_pre, y_pre])


# In[3]

count1_real=data_counts[Start:Stop:Step,0]
count2_real=data_counts[Start:Stop:Step,1]
count3_real=data_counts[Start:Stop:Step,2]
count4_real=data_counts[Start:Stop:Step,3]

count1_pred=np.zeros(len(count1_real))
count2_pred=np.zeros(len(count1_real))
count3_pred=np.zeros(len(count1_real))
count4_pred=np.zeros(len(count1_real))

for i in range(len(count1_pred)):
    count1_pred[i]=prediction[0][i][0]
    count2_pred[i]=prediction[1][i][0]
    count3_pred[i]=prediction[2][i][0]
    count4_pred[i]=prediction[3][i][0]

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.scatter((data_pos[Start:Stop:Step,0]-670.006)/1000, (data_pos[Start:Stop:Step,1]+199.486)/1000, count1_real,label='Real')
ax.scatter((data_pos[Start:Stop:Step,0]-670.006)/1000, (data_pos[Start:Stop:Step,1]+199.486)/1000, count1_pred,label='Pred')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('Count 1')
ax.set_title('Count 1')
ax.legend()


fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
ax.scatter((data_pos[Start:Stop:Step,0]-670.006)/1000, (data_pos[Start:Stop:Step,1]+199.486)/1000, count2_real,label='Real')
ax.scatter((data_pos[Start:Stop:Step,0]-670.006)/1000, (data_pos[Start:Stop:Step,1]+199.486)/1000, count2_pred,label='Pred')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('Count 2')
ax.set_title('Count 2')
ax.legend()


fig = plt.figure(3)
ax = fig.add_subplot(111, projection='3d')
ax.scatter((data_pos[Start:Stop:Step,0]-670.006)/1000, (data_pos[Start:Stop:Step,1]+199.486)/1000, count3_real,label='Real')
ax.scatter((data_pos[Start:Stop:Step,0]-670.006)/1000, (data_pos[Start:Stop:Step,1]+199.486)/1000, count3_pred,label='Pred')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('Count 3')
ax.set_title('Count 3')
ax.legend()



fig = plt.figure(4)
ax = fig.add_subplot(111, projection='3d')
ax.scatter((data_pos[Start:Stop:Step,0]-670.006)/1000, (data_pos[Start:Stop:Step,1]+199.486)/1000, count4_real,label='Real')
ax.scatter((data_pos[Start:Stop:Step,0]-670.006)/1000, (data_pos[Start:Stop:Step,1]+199.486)/1000, count4_pred,label='Pred')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('Count 4')
ax.set_title('Count 4')
ax.legend()