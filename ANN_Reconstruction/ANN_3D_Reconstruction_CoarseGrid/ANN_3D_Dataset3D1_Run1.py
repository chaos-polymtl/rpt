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
NUM_TP=1900379

# Feed file content
Feed=np.zeros([NUM_TP,9])

# File for the number of counts
filename_counts = 'interpolaited_counts_all_dataset3D1.txt'
data_counts = np.loadtxt(filename_counts, delimiter='\t')

for i in range(6):
    Feed[:,i] = data_counts[0:NUM_TP, i]

# File for the position x and y
filename_pos = 'x_y_robot_position_dataset3D1.txt'
data_pos = np.loadtxt(filename_pos, delimiter=',')


Feed[:,6] = (data_pos[0:NUM_TP, 0]-569.884)/1000
Feed[:,7] = (data_pos[0:NUM_TP, 1]+170.027)/1000
Feed[:,8] = (data_pos[0:NUM_TP, 2]-840.384)/1000
    
# Write the counts and position on a text file
with open("Feed.txt", "w") as file_output: 
    file_output.truncate(0)
    for row in Feed:
        line = '\t'.join([str(item) for item in row])
        file_output.write(f'{line}\n')

sampling_time=np.arange(1000)

counts=np.zeros(1000)
pos_rob=np.zeros(1000)
for i in range(len(counts)):
	counts[i]=(Feed[1458947+i,4]-min(Feed[1458947:1459947,4]))/(max(Feed[1458947:1459947,4])-min(Feed[1458947:1459947,4]))
	pos_rob[i]=(Feed[1458947+i,7]-min(Feed[1458947:1459947,7]))/(max(Feed[1458947:1459947,7])-min(Feed[1458947:1459947,7]))
	
plt.figure()
plt.scatter(sampling_time,counts,label='Counts')
plt.plot(sampling_time,pos_rob,label='Robot pos')
plt.legend()


#Define input features 
Count1= keras.Input(shape=(1,), name="Count1")
Count2= keras.Input(shape=(1,), name="Count2")
Count3= keras.Input(shape=(1,), name="Count3")
Count4= keras.Input(shape=(1,), name="Count4")
Count5= keras.Input(shape=(1,), name="Count5")
Count6= keras.Input(shape=(1,), name="Count6")

# Joining strings together end-to-end to create a new string to feed the model
# The new string is input layer
x = layers.concatenate([Count1,Count2,Count3,Count4,Count5,Count6])


#Define the hidden layers
hidden1 = layers.Dense(100, activation='elu')(x)
hidden2 = layers.Dense(100, activation='elu')(hidden1)
hidden3 = layers.Dense(100, activation='elu')(hidden2)
hidden4 = layers.Dense(100, activation='elu')(hidden3)
hidden5 = layers.Dense(100, activation='elu')(hidden4)
hidden6 = layers.Dense(100, activation='elu')(hidden5)


X = layers.Dense(1,activation='linear', name="xx")(hidden6)
Y = layers.Dense(1,activation='linear', name="yy")(hidden6)
Z = layers.Dense(1,activation='linear', name="zz")(hidden6)

model = keras.Model(inputs=[Count1,Count2,Count3,Count4,Count5,Count6],outputs=[X, Y, Z],)


keras.backend.set_epsilon(1)
model.compile(
    optimizer='Adamax',
    loss=['mse', 'mse','mse'],
    loss_weights=[1.0, 1.0,1.0],
    metrics=['MAE','mean_absolute_percentage_error']   
)

# Define the learning rate
LR=0.01
K.set_value(model.optimizer.learning_rate,LR)

pd_dat = pd.read_csv('Feed.txt', delimiter='\t')

# Extract the values from the dataframe
dataset = pd_dat.values

#Normalizing the input counts
X_raw=dataset[:,:6]
scaler_X = MinMaxScaler()
scaler_X.fit(X_raw)
X_scale= scaler_X.transform(X_raw)

# Training and testing the ANN
X_train, X_test, Y_train, Y_test = train_test_split(X_scale[:,:6],dataset[:,6:], test_size=0.2)


count1_train, count2_train, count3_train, count4_train, count5_train, count6_train = np.transpose(X_train)
count1_test, count2_test, count3_test, count4_test, count5_test, count6_test  = np.transpose(X_test)

with open('test.txt','w') as file:
	for i in range(len(Y_train)):
		file.write(str(Y_train[i,:])+'\n')
	
x_train, y_train, z_train = Y_train[:,0], Y_train[:,1], Y_train[:,2]
x_test, y_test, z_test = Y_test[:,0], Y_test[:,1], Y_test[:,2]


inputs_train=[count1_train, count2_train, count3_train, count4_train, count5_train, count6_train]
outputs_train=[x_train, y_train, z_train]


history=model.fit(inputs_train,outputs_train,
                 validation_split=0.2,
                 epochs=5000,
                 batch_size=20000,
                 )
print(history.history)


result=model.evaluate([count1_test, count2_test, count3_test, count4_test, count5_test, count6_test],[x_test,y_test,z_test],verbose=2)
print(result)

# Plot of the loss
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='validation')
pyplot.legend()


# In[13]:

# Array of the counts used for the prediction of the spiral
Pred = np.zeros([15500,6])

for i in range(6):
    Pred[:,i] = data_counts[-15501:-1, i] 


X_pre=Pred[:,:6]
scaler_X_pre = MinMaxScaler()
scaler_X_pre.fit(X_pre)
X_scale_pre= scaler_X.transform(X_pre)
count1_pre, count2_pre, count3_pre, count4_pre, count5_pre, count6_pre=np.transpose(X_scale_pre)
prediction=model.predict([count1_pre, count2_pre, count3_pre, count4_pre, count5_pre, count6_pre])

# In[14]: 

# Spiral Reconstruction

# Real position of the robot during the spiral
x_real=(data_pos[-15501:-1, 0]-569.884)/1000
y_real=(data_pos[-15501:-1, 1]+170.027)/1000  
z_real=(data_pos[-15501:-1, 2]-840.384)/1000  

ax=plt.figure().add_subplot(projection='3d')
ax.plot(x_real, y_real, z_real, label='Real')
ax.legend()

# Convert the data to a NumPy array
x_pred=np.zeros(len(x_real))
y_pred=np.zeros(len(y_real))
z_pred=np.zeros(len(z_real))


for i in range(len(x_pred)):
    x_pred[i]=prediction[0][i]
    y_pred[i]=prediction[1][i]
    z_pred[i]=prediction[2][i]
    
    
ax.plot(x_pred,y_pred, z_pred,label='Prediction')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')


Erreur_x=np.abs(x_pred-x_real)
MAE_x=Erreur_x.mean()

Erreur_y=np.abs(y_pred-y_real)
MAE_y=Erreur_y.mean()

Erreur_z=np.abs(z_pred-z_real)
MAE_z=Erreur_z.mean()

print('MAE_x = ', MAE_x, '\n', 'MAE_y', MAE_y,'\n', 'MAE_z', MAE_z)

# In[15]
# Velocities

N=9000 # Number of points for the spiral
n_pts=125 # Number of points to skip for the centered second order derivative on the predicted spiral
n_pts_real=1 # Number of points to skip for the centered second order derivative on the actual spiral

# to make sure the velocity vector is the right size
if N/n_pts %1 >=0.5:
    v_x_real=np.zeros(round(N/n_pts-0.5)-1)
    v_y_real=np.zeros(round(N/n_pts-0.5)-1)
    v_z_real=np.zeros(round(N/n_pts-0.5)-1)
    v_x_pred=np.zeros(round(N/n_pts-0.5)-1)
    v_y_pred=np.zeros(round(N/n_pts-0.5)-1)
    v_z_pred=np.zeros(round(N/n_pts-0.5)-1)
else:
    v_x_real=np.zeros(round(N/n_pts-0.5)-2)
    v_y_real=np.zeros(round(N/n_pts-0.5)-2)
    v_z_real=np.zeros(round(N/n_pts-0.5)-2)
    v_x_pred=np.zeros(round(N/n_pts-0.5)-2)
    v_y_pred=np.zeros(round(N/n_pts-0.5)-2)
    v_z_pred=np.zeros(round(N/n_pts-0.5)-2)


for i in range(0,len(v_x_pred)):
    if (i+1)*n_pts+n_pts>=N: # To make sure the size of the position vector is not exceeded
        print('error')
    else:
        # Centered second ordrer derivative
        v_x_pred[i]=(x_pred[(i+1)*n_pts+n_pts]-x_pred[(i+1)*n_pts-n_pts])/(2*0.01*n_pts)
        v_y_pred[i]=(y_pred[(i+1)*n_pts+n_pts]-y_pred[(i+1)*n_pts-n_pts])/(2*0.01*n_pts)
        v_z_pred[i]=(z_pred[(i+1)*n_pts+n_pts]-z_pred[(i+1)*n_pts-n_pts])/(2*0.01*n_pts)
        v_x_real[i]=(x_real[(i+1)*n_pts+n_pts_real]-x_real[(i+1)*n_pts-n_pts_real])/(2*0.01*n_pts_real)
        v_y_real[i]=(y_real[(i+1)*n_pts+n_pts_real]-y_real[(i+1)*n_pts-n_pts_real])/(2*0.01*n_pts_real)
        v_z_real[i]=(z_real[(i+1)*n_pts+n_pts_real]-z_real[(i+1)*n_pts-n_pts_real])/(2*0.01*n_pts_real)
        
# Magnitude of the overall velocity
v_real=np.sqrt(np.square(v_x_real)+np.square(v_y_real)+np.square(v_z_real))	
v_pred=np.sqrt(np.square(v_x_pred)+np.square(v_y_pred)+np.square(v_z_pred))


# Velocity vector's angle in the x-y plane
Theta_real=np.arctan2(v_y_real,v_x_real)
Theta_pred=np.arctan2(v_y_pred,v_x_pred)


# Magnitude of the velocity in the x-y plane
v_plan_real=np.sqrt(np.square(v_x_real)+np.square(v_y_real))
v_plan_pred=np.sqrt(np.square(v_x_pred)+np.square(v_y_pred))

# Velocity vector's angle according to the z-axis
Phi_real=np.arctan2(v_z_real, v_plan_real)
Phi_pred=np.arctan2(v_z_pred, v_plan_pred)

# Plot of the velocity on the spiral position reconstruction plot
for i in range(len(v_pred)):
    if i*n_pts>N:
        break
    else:
        plt.quiver(x_pred[(i+1)*n_pts],y_pred[(i+1)*n_pts], z_pred[(i+1)*n_pts],v_x_pred[i],v_y_pred[i], v_z_pred[i])
        
Erreur_vx=np.zeros(len(v_x_pred))
for i in range(len(Erreur_vx)):
    if v_x_real[i]==0: # to make sure there are no division by 0
        Erreur_vx[i]=np.abs(np.divide(v_x_pred[i]-v_x_real[i],v_x_pred[i]))
    else:
        Erreur_vx[i]=np.abs(np.divide(v_x_pred[i]-v_x_real[i],v_x_real[i]))   
MAE_vx=Erreur_vx.mean()


Erreur_vy=np.zeros(len(v_y_pred))
for i in range(len(Erreur_vy)):
    if v_y_real[i]==0:
        Erreur_vy[i]=np.abs(np.divide(v_y_pred[i]-v_y_real[i],v_y_pred[i]))
    else:
        Erreur_vy[i]=np.abs(np.divide(v_y_pred[i]-v_y_real[i],v_y_real[i]))
MAE_vy=Erreur_vy.mean()


Erreur_vz=np.zeros(len(v_z_pred))
for i in range(len(Erreur_vz)):
    if v_z_real[i]==0:
        Erreur_vz[i]=np.abs(np.divide(v_z_pred[i]-v_z_real[i],v_z_pred[i]))
    else:
        Erreur_vz[i]=np.abs(np.divide(v_z_pred[i]-v_z_real[i],v_z_real[i]))
MAE_vz=Erreur_vz.mean()


Erreur_v=np.zeros(len(v_pred))
for i in range(len(Erreur_v)):
    if v_real[i]==0:
        Erreur_v[i]=np.abs(np.divide(v_pred[i]-v_real[i],v_pred[i]))
    else:
        Erreur_v[i]=np.abs(np.divide(v_pred[i]-v_real[i],v_real[i]))
MAE_v=Erreur_v.mean()

Erreur_theta=np.abs(Theta_pred-Theta_real)
MAE_theta=Erreur_theta.mean()

Erreur_phi=np.abs(Phi_pred-Phi_real)
MAE_phi=Erreur_phi.mean()

print('MAE_vx = ', MAE_vx, '\n', 'MAE_vy = ', MAE_vy, '\n','MAE_vz = ', MAE_vz, '\n', 'MAE_v = ', MAE_v, '\n MAE_theta = ', MAE_theta, '\n MAE_phi = ', MAE_phi)

# In[16]: 
	
# Line pattern x reconstruction

# Real line points
plt.figure()
plt.plot(Feed[347000:348500,6],Feed[347000:348500,7],label='Real')

# Line prediction
X_pre=Feed[347000:348500,:6]
scaler_X_pre = MinMaxScaler()
scaler_X_pre.fit(X_pre)
X_scale_pre= scaler_X.transform(X_pre)
count1_pre, count2_pre, count3_pre,count4_pre, count5_pre, count6_pre=np.transpose(X_scale_pre)
prediction=model.predict([count1_pre, count2_pre, count3_pre,count4_pre, count5_pre, count6_pre])

plt.plot(prediction[0][:],prediction[1][:],label='Pred')
plt.title('End pattern')
plt.legend()

#Plot of the error in x along the line
sampling_time=np.arange(1500)

plt.figure()
plt.plot(sampling_time,Feed[347000:348500,6],label='Real')
plt.plot(sampling_time,prediction[0][:],label='Pred')
plt.xlabel('Sampling time for the line')
plt.ylabel('x')
plt.legend()

# Calculating the error on the lag
ERRORLAG=prediction[0][:]-Feed[347000:348500,6]
ERRORLAG=ERREURLAG.mean()
print('\n Error lag x : ', ERRORLAG)


# In[16]
	
# Line pattern y reconstruction

# Real line points
plt.figure()
plt.plot(Feed[447000:448500,6],Feed[447000:448500,7],label='Real')

# Line prediction
X_pre=Feed[447000:448500,:6]
scaler_X_pre = MinMaxScaler()
scaler_X_pre.fit(X_pre)
X_scale_pre= scaler_X.transform(X_pre)
count1_pre, count2_pre, count3_pre,count4_pre, count5_pre, count6_pre=np.transpose(X_scale_pre)
prediction=model.predict([count1_pre, count2_pre, count3_pre,count4_pre, count5_pre, count6_pre])

plt.plot(prediction[0][:],prediction[1][:],label='Pred')
plt.title('End pattern')
plt.legend()

# Plot of the error in y along the line
sampling_time=np.arange(1500)

plt.figure()
plt.plot(sampling_time,Feed[447000:448500,7],label='Real')
plt.plot(sampling_time,prediction[1][:],label='Pred')
plt.xlabel('Sampling time for the line')
plt.ylabel('y')
plt.legend()


ERRORLAG=prediction[1][:]-Feed[447000:448500,7]
ERRORLAG=ERREURLAG.mean()
print('\n Erreur lag y : ', ERRORLAG)


# Grid reconstruction

# Real grid points
plt.figure()
plt.plot(Feed[-155900:-1,4],Feed[-155900:-1,5],label='Real')

# Grid prediction
X_pre=Feed[-155900:-1,:4]

scaler_X_pre = MinMaxScaler()
scaler_X_pre.fit(X_pre)
X_scale_pre= scaler_X.transform(X_pre)
count1_pre, count2_pre, count3_pre,count4_pre=np.transpose(X_scale_pre)
prediction=model.predict([count1_pre, count2_pre, count3_pre,count4_pre])

plt.plot(prediction[0][:],prediction[1][:],label='Pred')
plt.title('Grid pattern')
plt.legend()

plt.show()

