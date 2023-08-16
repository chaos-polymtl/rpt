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
NUM_TP=390000

# Feed file content
Feed=np.zeros([NUM_TP,6])

# File for the number of counts
filename_counts = 'interpolaited_counts_all_datasetA.txt'
data_counts = np.loadtxt(filename_counts, delimiter='\t')

for i in range(4):
    Feed[:,i] = data_counts[0:NUM_TP, i]

# File for the position x and y
filename_pos = 'x_y_robot_position_datasetA.txt'
data_pos = np.loadtxt(filename_pos, delimiter=',')


Feed[:,4] = (data_pos[0:NUM_TP, 0]-670)/1000
Feed[:,5] = (data_pos[0:NUM_TP, 1]--199.494)/1000
    
# Write the counts and position on a text file
with open("Feed.txt", "w") as file_output: 
    file_output.truncate(0)
    for row in Feed:
        line = '\t'.join([str(item) for item in row])
        file_output.write(f'{line}\n')


#Define input features 
Count1= keras.Input(shape=(1,), name="Count1")
Count2= keras.Input(shape=(1,), name="Count2")
Count3= keras.Input(shape=(1,), name="Count3")
Count4= keras.Input(shape=(1,), name="Count4")


# Joining strings together end-to-end to create a new string to feed the model
# The new string is input layer
x = layers.concatenate([Count1,Count2,Count3,Count4])


#Define the hidden layers
hidden1 = layers.Dense(60, activation='elu')(x)
hidden2 = layers.Dense(60, activation='elu')(hidden1)
hidden3 = layers.Dense(60, activation='elu')(hidden2)

# Define the output layers
X = layers.Dense(1,activation='linear', name="xx")(hidden3)
Y = layers.Dense(1,activation='linear', name="yy")(hidden3)


model = keras.Model(inputs=[Count1,Count2,Count3,Count4],outputs=[X, Y],)


keras.backend.set_epsilon(1)
model.compile(
    optimizer='Adam',
    loss=['mse', 'mse'],
    loss_weights=[1.0, 1.0],
    metrics=['MAE','mean_absolute_percentage_error']   
)

# Define the learning rate
LR=0.002
K.set_value(model.optimizer.learning_rate,LR)

pd_dat = pd.read_csv('Feed.txt', delimiter='\t')

# Extract the values from the dataframe
dataset = pd_dat.values

#Normalizing the input counts
X_raw=dataset[:,:4]
scaler_X = MinMaxScaler()
scaler_X.fit(X_raw)
X_scale= scaler_X.transform(X_raw)


# Training and testing the Neural Network
X_train, X_test, Y_train, Y_test=train_test_split(X_scale[:,:4],dataset[:,4:], test_size=0.2)

count1_train, count2_train, count3_train,count4_train=np.transpose(X_train)
count1_test, count2_test, count3_test,count4_test=np.transpose(X_test)

x_train, y_train= Y_train[:,0], Y_train[:,1]
x_test, y_test= Y_test[:,0], Y_test[:,1]

inputs_train=[count1_train, count2_train, count3_train,count4_train]
outputs_train=[x_train,y_train]


history=model.fit(inputs_train,outputs_train,
                validation_split=0.2,
                 epochs=1000,
                 batch_size=80000,
                 )
print(history.history)

result=model.evaluate([count1_test, count2_test, count3_test,count4_test],[x_test,y_test],verbose=2)
print(result)

# Plot of the loss
pyplot.subplot(211)
pyplot.title('Loss')
plt.semilogy(history.history['loss'], label='train')
plt.semilogy(history.history['val_loss'], label='validation')
plt.xscale("log")
pyplot.legend()


# In[13]:

# Array of the counts used for the prediction of the spiral
Pred = np.zeros([15500,4])

for i in range(4):
    Pred[:,i] = data_counts[-15501:-1, i] 

# Writes the prediction in a .csv file
with open("Prediction.csv",'w', newline='') as file:
    file.truncate(0)
    writer = csv.writer(file)
    writer.writerows(Pred)
    

pd_pre=pd.read_csv('Prediction.csv')
datasetpre=pd_pre.values


X_pre=datasetpre[:,:4]
scaler_X_pre = MinMaxScaler()
scaler_X_pre.fit(X_pre)
X_scale_pre= scaler_X.transform(X_pre)
count1_pre, count2_pre, count3_pre,count4_pre=np.transpose(X_scale_pre)
prediction=model.predict([count1_pre, count2_pre, count3_pre,count4_pre])


arr=np.asarray(prediction[0])
pd.DataFrame(arr).to_csv('x.csv')

arr=np.asarray(prediction[1])
pd.DataFrame(arr).to_csv('y.csv')


# In[14]: 

# Real position of the robot during the spiral
x_real=(data_pos[-15501:-1, 0]-670.006)/1000
y_real=(data_pos[-15501:-1, 1]+199.486)/1000  

plt.figure()
plt.plot(x_real,y_real,label='Real')


# Read the CSV file
with open('x.csv', 'r') as file:
    reader = csv.reader(file)
    x_p = list(reader)


# Read the CSV file
with open('y.csv', 'r') as file:
    reader = csv.reader(file)
    y_p = list(reader)


# Convert the data to a NumPy array
x_pred=np.zeros(len(x_p))
y_pred=np.zeros(len(y_p))
for i in range(len(x_pred)):
    x_pred[i]=x_p[i][1]
    y_pred[i]=y_p[i][1]

plt.plot(x_pred,y_pred,label='Prediction')
plt.xlabel('x')
plt.ylabel('y')

Erreur_x = np.square(x_pred-x_real)
RMSE_x = np.sqrt(sum(Erreur_x)/len(Erreur_x))

STD_x = np.sqrt(np.mean(abs(Erreur_x - Erreur_x.mean())**2))

Erreur_y = np.square(y_pred-y_real)
RMSE_y = np.sqrt(sum(Erreur_y)/len(Erreur_y))

STD_y = np.sqrt(np.mean(abs(Erreur_y - Erreur_y.mean())**2))

Erreur_x=np.abs(x_pred-x_real)
MAE_x=Erreur_x.mean()

Erreur_y=np.abs(y_pred-y_real)
MAE_y=Erreur_y.mean()

print('RMSE_x = ', RMSE_x, '\n', 'STD_x = ', STD_x, '\n', 'RMSE_y = ',RMSE_y,'\n', 'STD_y = ',STD_y, '\n', 'MAE_x = ', MAE_x, '\n', 'MAE_y', MAE_y)

# Velocities

n_pts=300 # Number of points between each point taken for the velocity's approximation

# Real velocities
v_x_real=np.zeros(round(N/n_pts-0.5))
v_y_real=np.zeros(round(N/n_pts-0.5))
v_x_pred=np.zeros(round(N/n_pts-0.5))
v_y_pred=np.zeros(round(N/n_pts-0.5))

for i in range(len(v_x_pred)):
    if (i+1)*n_pts-1>N:
        break
    else:
        v_x_pred[i]=(x_pred[(i+1)*n_pts-1]-x_pred[i*n_pts])/(2*0.01)
        v_y_pred[i]=(y_pred[(i+1)*n_pts-1]-y_pred[i*n_pts])/(2*0.01)
        v_x_real[i]=(x_real[(i+1)*n_pts-1]-x_real[i*n_pts])/(2*0.01)        
        v_y_real[i]=(y_real[(i+1)*n_pts-1]-y_real[i*n_pts])/(2*0.01)

v_real=np.sqrt(np.square(v_x_real)+np.square(v_y_real))	
v_pred=np.sqrt(np.square(v_x_pred)+np.square(v_y_pred))

# Real velocity direction
Theta_real=np.arctan2(v_y_real,v_x_real)
Theta_pred=np.arctan2(v_y_pred,v_x_pred)

for i in range(len(v_pred)):
    if i*n_pts>N:
        break
    else:
    # Plot of the velocity magnitude and direction 
        plt.arrow(x_pred[i*n_pts],y_pred[i*n_pts],v_x_pred[i]*0.01,v_y_pred[i]*0.01)
​
Erreur_vx=np.abs(v_x_pred-v_x_real)
MAE_vx=Erreur_vx.mean()

Erreur_vy=np.abs(v_y_pred-v_y_real)
MAE_vy=Erreur_vy.mean()

Erreur_v=np.abs(v_pred-v_real)
MAE_v=Erreur_v.mean()

Erreur_theta=np.abs(Theta_pred-Theta_real)
MAE_theta=Erreur_theta.mean()

print('MAE_vx = ', MAE_vx, '\n', 'MAE_vy = ', MAE_vy, '\n', 'MAE_v = ', MAE_v, '\n MAE_theta = ', MAE_theta)

​
# In[15]: 
	
# Line reconstruction

# Real line points
plt.figure()
plt.plot(Feed[-25000:-125800,4],Feed[-127300:-125800,5],label='Real')

# Line prediction
X_pre=Feed[-127300:-125800,:4]
scaler_X_pre = MinMaxScaler()
scaler_X_pre.fit(X_pre)
X_scale_pre= scaler_X.transform(X_pre)
count1_pre, count2_pre, count3_pre,count4_pre=np.transpose(X_scale_pre)
prediction=model.predict([count1_pre, count2_pre, count3_pre,count4_pre])

plt.plot(prediction[0][:],prediction[1][:],label='Pred')
plt.title('End pattern')
plt.legend()

# Plot of the error along the line - To fix the lag
sampling_time=np.arange(1500)

plt.figure()
plt.plot(sampling_time,Feed[-127300:-125800,4],label='Real')
plt.plot(sampling_time,prediction[0][:],label='Pred')
plt.xlabel('Sampling time for the line')
plt.ylabel('x')
plt.legend()


# In[16]

# Grid reconstruction

# Real grid points
plt.figure()
plt.plot(Feed[20000:180000,4],Feed[20000:180000,5],label='Real')

# Grid prediction
X_pre=Feed[20000:180000,:4]
scaler_X_pre = MinMaxScaler()
scaler_X_pre.fit(X_pre)
X_scale_pre= scaler_X.transform(X_pre)
count1_pre, count2_pre, count3_pre,count4_pre=np.transpose(X_scale_pre)
prediction=model.predict([count1_pre, count2_pre, count3_pre,count4_pre])

plt.plot(prediction[0][:],prediction[1][:],label='Pred')
plt.title('Grid pattern')
plt.legend()

plt.show()
  



