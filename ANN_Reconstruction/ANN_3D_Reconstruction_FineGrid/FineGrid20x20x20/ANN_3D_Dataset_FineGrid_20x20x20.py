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
NUM_TP=3498524

# Feed file content
Feed=np.zeros([NUM_TP,9])

# File for the number of counts
filename_counts = 'interpolaited_counts_all_dataset3D2.txt'
data_counts = np.loadtxt(filename_counts, delimiter='\t')

for i in range(6):
    Feed[:,i] = data_counts[0:NUM_TP, i]

# File for the position x and y
filename_pos = 'x_y_robot_position_dataset3D2.txt'
data_pos = np.loadtxt(filename_pos, delimiter=',')


Feed[:,6] = (data_pos[0:NUM_TP, 0]-570.004)/1000
Feed[:,7] = (data_pos[0:NUM_TP, 1]+169.990)/1000
Feed[:,8] = (data_pos[0:NUM_TP, 2]-840.497)/1000

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
Count5= keras.Input(shape=(1,), name="Count5")
Count6= keras.Input(shape=(1,), name="Count6")

# Joining strings together end-to-end to create a new string to feed the model
# The new string is input layer
x = layers.concatenate([Count1,Count2,Count3,Count4,Count5,Count6])


#Define the hidden layers
hidden1 = layers.Dense(128, activation='relu')(x)
hidden2 = layers.Dense(64, activation='relu')(hidden1)
hidden3 = layers.Dense(32, activation='relu')(hidden2)
hidden4 = layers.Dense(16, activation='relu')(hidden3)
hidden5 = layers.Dense(8, activation='relu')(hidden4)

X = layers.Dense(1,activation='linear', name="xx")(hidden5)
Y = layers.Dense(1,activation='linear', name="yy")(hidden5)
Z = layers.Dense(1,activation='linear', name="zz")(hidden5)

model = keras.Model(inputs=[Count1,Count2,Count3,Count4,Count5,Count6],outputs=[X, Y, Z],)


keras.backend.set_epsilon(1)
model.compile(
    optimizer='Adam',
    loss=['mse', 'mse','mse'],
    loss_weights=[1.0, 1.0,1.0],
    metrics=['MAE','mean_absolute_percentage_error']   
)

# Define the learning rate
LR=0.001
K.set_value(model.optimizer.learning_rate,LR)

pd_dat = pd.read_csv('Feed.txt', delimiter='\t')

# Extract the values from the dataframe
dataset = pd_dat.values

#Normalizing the input counts
X_raw=dataset[:,:6]
scaler_X = MinMaxScaler()
scaler_X.fit(X_raw)
X_scale= scaler_X.transform(X_raw)


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
                 validation_split=0.3,
                 epochs=100,
                 batch_size=50000,
                 )
print(history.history)


result=model.evaluate([count1_test, count2_test, count3_test, count4_test, count5_test, count6_test],[x_test,y_test,z_test],verbose=2)
print(result)


pyplot.subplot(211)
pyplot.title('Loss RUN 4')
plt.semilogy(history.history['loss'], label='train')
plt.semilogy(history.history['val_loss'], label='validation')
plt.xscale("log")
pyplot.legend()


# In[13]:

# Array of the counts used for the prediction of the spiral
Pred = data_counts[-13301:-1, :] 


X_pre=Pred[:,:6]
scaler_X_pre = MinMaxScaler()
scaler_X_pre.fit(X_pre)
X_scale_pre= scaler_X.transform(X_pre)
count1_pre, count2_pre, count3_pre, count4_pre, count5_pre, count6_pre=np.transpose(X_scale_pre)
prediction=model.predict([count1_pre, count2_pre, count3_pre, count4_pre, count5_pre, count6_pre])

# In[14]: 

# Real position of the robot during the spiral
x_real=(data_pos[-13001:-1, 0]-570.004)/1000
y_real=(data_pos[-13001:-1, 1]+169.990)/1000  
z_real=(data_pos[-13001:-1, 2]-840.497)/1000  

ax=plt.figure().add_subplot(projection='3d')
ax.plot(x_real, y_real, z_real, label='Real')


# Convert the data to a NumPy array
x_pred=np.zeros(len(x_real))
y_pred=np.zeros(len(y_real))
z_pred=np.zeros(len(z_real))


for i in range(len(x_pred)):
    x_pred[i]=prediction[0][i][0]
    y_pred[i]=prediction[1][i][0]
    z_pred[i]=prediction[2][i][0]
    
    
ax.plot(x_pred,y_pred, z_pred,label='Prediction')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Spiral RUN 5')
ax.legend()

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
	
# Lign pattern x

plt.figure()
plt.plot(Feed[348200:349800,6],Feed[348200:349800,7],label='Real')


X_pre=Feed[348200:349800,:6]
scaler_X_pre = MinMaxScaler()
scaler_X_pre.fit(X_pre)
X_scale_pre= scaler_X.transform(X_pre)
count1_pre, count2_pre, count3_pre,count4_pre, count5_pre, count6_pre=np.transpose(X_scale_pre)
prediction=model.predict([count1_pre, count2_pre, count3_pre,count4_pre, count5_pre, count6_pre])

plt.plot(prediction[0][:],prediction[1][:],label='Pred')
plt.title('Line x pattern RUN 5')
plt.legend()

sampling_time=np.arange(1600)

plt.figure()
plt.plot(sampling_time,Feed[348200:349800,6],label='Real')
plt.plot(sampling_time,prediction[0][:],label='Pred')
plt.xlabel('Sampling time for the line RUN 5')
plt.ylabel('x')
plt.legend()


ERREURLAG=prediction[0][:]-Feed[348200:349800,6]
ERREURLAG=ERREURLAG.mean()
print('\n Erreur lag x : ', ERREURLAG)


# In[16]
	
# Lign pattern y

plt.figure()
plt.plot(Feed[447750:449450,6],Feed[447750:449450,7],label='Real')


X_pre=Feed[447750:449450,:6]
scaler_X_pre = MinMaxScaler()
scaler_X_pre.fit(X_pre)
X_scale_pre= scaler_X.transform(X_pre)
count1_pre, count2_pre, count3_pre,count4_pre, count5_pre, count6_pre=np.transpose(X_scale_pre)
prediction=model.predict([count1_pre, count2_pre, count3_pre,count4_pre, count5_pre, count6_pre])

plt.plot(prediction[0][:],prediction[1][:],label='Pred')
plt.title('Line y pattern RUN 5')
plt.legend()

sampling_time=np.arange(1700)

plt.figure()
plt.plot(sampling_time,Feed[447750:449450,7],label='Real')
plt.plot(sampling_time,prediction[1][:],label='Pred')
plt.xlabel('Sampling time for the line RUN 5')
plt.ylabel('y')
plt.legend()


ERREURLAG=prediction[1][:]-Feed[447750:449450,7]
ERREURLAG=ERREURLAG.mean()
print('\n Erreur lag y : ', ERREURLAG)


# Grid

# Random grid

plt.figure()
plt.plot(Feed[1000000:1162000,6],Feed[1000000:1162000,7],label='Real')

X_pre=Feed[1000000:1162000,:6]
scaler_X_pre = MinMaxScaler()
scaler_X_pre.fit(X_pre)
X_scale_pre= scaler_X.transform(X_pre)
count1_pre, count2_pre, count3_pre,count4_pre, count5_pre, count6_pre=np.transpose(X_scale_pre)
prediction=model.predict([count1_pre, count2_pre, count3_pre,count4_pre,count5_pre,count6_pre])

plt.plot(prediction[0][:],prediction[1][:],label='Pred')
plt.title('Grid pattern (random grid)')
plt.legend()

x_pred=np.zeros(len(Feed[1000000:1162000,6]))
y_pred=np.zeros(len(Feed[1000000:1162000,7]))

for i in range(len(x_pred)):
    x_pred[i]=prediction[0][i][0]
    y_pred[i]=prediction[1][i][0]

Erreur_x=np.abs(x_pred-Feed[1000000:1162000,6])
MAE_x=Erreur_x.mean()

Erreur_y=np.abs(y_pred-Feed[1000000:1162000,7])
MAE_y=Erreur_y.mean()

print('MAE_x_grid = ', MAE_x, '\n', 'MAE_y_grid', MAE_y)

# First grid

plt.figure()
plt.plot(Feed[0:162000,6],Feed[0:162000,7],label='Real')

X_pre=Feed[0:162000,:6]
scaler_X_pre = MinMaxScaler()
scaler_X_pre.fit(X_pre)
X_scale_pre= scaler_X.transform(X_pre)
count1_pre, count2_pre, count3_pre,count4_pre, count5_pre, count6_pre=np.transpose(X_scale_pre)
prediction=model.predict([count1_pre, count2_pre, count3_pre,count4_pre,count5_pre,count6_pre])

plt.plot(prediction[0][:],prediction[1][:],label='Pred')
plt.title('Grid pattern (first grid)')
plt.legend()

x_pred=np.zeros(len(Feed[0:162000,6]))
y_pred=np.zeros(len(Feed[0:162000,7]))

for i in range(len(x_pred)):
    x_pred[i]=prediction[0][i][0]
    y_pred[i]=prediction[1][i][0]

Erreur_x=np.abs(x_pred-Feed[0:162000,6])
MAE_x=Erreur_x.mean()

Erreur_y=np.abs(y_pred-Feed[0:162000,7])
MAE_y=Erreur_y.mean()

print('MAE_x_grid = ', MAE_x, '\n', 'MAE_y_grid', MAE_y)

plt.show()

