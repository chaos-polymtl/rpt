# -*- coding: utf-8 -*-
"""

Created on Fri Aug 25 09:56:25 2023

@author: Thomas MONATTE

"""
from scipy.stats import qmc 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math 




' This code aims to generates points within a cylinder using Latin Hypercube Sampling (LHS). It generates'
' three files: x1_values_cyl, x2_values_cyl, and x3_values_cyl, which are x, y , and z positions in the cylinder.


########################       Parameters for the code       ######################## 

# Samples number
num_samples=1000

# Dimension of the geometry
dim=3

# Seed for the random numbers
seed=1

# Dimensions of the geometry (in mm)

R_tank=100         # radius of the tank
H_tank=200        # height of the tank

lim=10
R=R_tank-lim        # radius for the points
H=H_tank-lim       # height for the points

# Particle radius (in mm) to calculate the volume
r=3

# Plot in 3D
title='LHS Cylinder ; 1000 pts'

# x1, x2, x3 are the positions of the particle 
# respectively equals to x, y and z


###########################     Function for LHS cylinder      ###########################    
    
def get_latin_hypercube_cylinder(bounds, num_samples, seed):
    # Define the cylinder bounds (height and radius)
    H_bounds = bounds['height']
    R_bounds = bounds['radius']

    # Generate Latin Hypercube Samples
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    samples = sampler.random(num_samples)

    # Scale samples to cylinder bounds
    H_samples = (samples[:, 0] * (H_bounds[1] - H_bounds[0]) + H_bounds[0])*-1 #-1 becuase in robot coardinate we have all z lower than 0
    R_samples = samples[:, 1] * (R_bounds[1]**2 - R_bounds[0]**2) + R_bounds[0]**2
    R_samples=np.sqrt(R_samples)

   
    plt.scatter(R_samples,H_samples)
    plt.title('Samples in a Latin Hypercube')
    plt.xlabel('R_tank (in mm)')
    plt.ylabel('H_tank (in mm)')
    plt.show()

    # Convert to Cartesian coordinates
    theta = np.random.uniform(0, 2 * np.pi, num_samples)
    x_samples = (R_samples * np.cos(theta))+100
    y_samples = R_samples * np.sin(theta)

    # Create DataFrame with cylinder samples
    data = {'x': x_samples, 'y': y_samples, 'z': H_samples}
    return pd.DataFrame(data)

if dim==3:
    bounds={'radius':[0,R],'height':[lim,H]}  
if dim==2:
    bounds={'radius':[0,R],'height':[0,0]}  

cylinder_samples = get_latin_hypercube_cylinder(bounds, num_samples, seed=seed)
 
if dim==2:
    x1_values = cylinder_samples['x'].tolist()
    x2_values = cylinder_samples['y'].tolist()
if dim==3:
    x1_values = cylinder_samples['x'].tolist()
    x2_values = cylinder_samples['y'].tolist()
    x3_values = cylinder_samples['z'].tolist()
    
    
#######################     Files for positions      ########################### 


# In 3D, open three files with respectively all the x, y and z positions 

if dim==3:
    with open('x1_values_cyl.txt','w') as file:
        for element in x1_values:
            file.write(str(element)+',\n')
            
    with open('x2_values_cyl.txt','w') as file:
        for element in x2_values:
            file.write(str(element)+',\n')
    
    with open('x3_values_cyl.txt','w') as file:
        for element in x3_values:
            file.write(str(element)+',\n')



###########################     Plot Figures      ###########################

alpha=np.linspace(0, 2*np.pi, 100)
radius=R_tank
x=radius*np.cos(alpha)
y=radius*np.sin(alpha)+100

alpha=np.linspace(0, 2*np.pi, 100)
radius=R
x2=radius*np.cos(alpha)
y2=radius*np.sin(alpha)+100

"""
# PLOT 2D
if dim==2:
    plt.scatter(x1_values,x2_values,s=0.1)
    plt.plot(x,y,'b',x2,y2,'r--')
    plt.legend(['Points for cylinder','Boundary of tank','Boundary for points'],loc='lower right')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Hyper Cube Latin 2D')
    plt.show()
"""
# Plot 3D
if dim==3:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
         
    ax.scatter(x1_values, x2_values, x3_values)    
    ax.set_xlabel('X')
    ax.set_xlim(0,R+100)
    ax.set_ylabel('Y')
    ax.set_ylim(-R,R)
    ax.set_zlabel('Z')
    ax.set_zlim(-H,0)
    ax.set_title(title)
    plt.show()
    
    
    plt.scatter(x2_values,x1_values)
    plt.plot(x,y,'b',x2,y2,'r--')
    plt.legend(['Points for cylinder','Boundary of tank','Boundary for points'],loc='lower right')
    plt.xlabel('Y')
    plt.xlim(R+20,-R-20)
    plt.ylabel('X')
    plt.ylim(R+R+40,-R+R-20)
    plt.title('LHS upper view')
    plt.show()
    
    
    # To visualize the boundary for the tank
    ty_verti=np.linspace(0,H_tank,100)
    tx_verti_1=[R_tank for k in range(len(ty_verti))]
    tx_verti_2=[-R_tank for k in range(len(ty_verti))]
    
    tx_horiz=np.linspace(-R_tank,R_tank,100)
    ty_horiz_1=[0 for k in range(len(tx_horiz))]
    ty_horiz_2=[H_tank for k in range(len(tx_horiz))]
    
    # To visualize the boundary for the points
    y_verti=np.linspace(0+lim,H_tank-lim,100)
    x_verti_1=[R_tank-lim for k in range(len(y_verti))]
    x_verti_2=[-R_tank+lim for k in range(len(y_verti))]
    
    x_horiz=np.linspace(-R_tank+lim,R_tank-lim,100)
    y_horiz_1=[0+lim for k in range(len(x_horiz))]
    y_horiz_2=[H_tank-lim for k in range(len(x_horiz))]
    
    
    """
    plt.scatter(x2_values,x3_values)
    plt.plot(tx_verti_1,ty_verti,'b') 
    plt.plot(x_verti_1,y_verti,'r--') 
    plt.plot(tx_verti_2,ty_verti,'b')
    plt.plot(x_verti_2,y_verti,'r--')
    plt.plot(tx_horiz,ty_horiz_1,'b',tx_horiz,ty_horiz_2,'b')
    plt.plot(x_verti_1,y_verti,'r--',x_verti_2,y_verti,'r--')
    plt.plot(x_horiz,y_horiz_1,'r--',x_horiz,y_horiz_2,'r--')
    plt.legend(['Points for cylinder','Boundary of tank','Boundary for points'],loc='lower right')
    plt.xlabel('Y')
    plt.xlim(-R-20,R+20)
    plt.ylabel('Z')
    plt.ylim(-10,H+20)
    plt.title('LHS side view')
    plt.show()
    """
    
    
    
    
###########################     Euclidean distance      ########################### 

all_list_dist=[]    # every distances between points, including peers and 0
list_dist=[]        # every distance between points, including 0 and without peers
list_mindist=[]     # the lowest distance for each points

# Plot 2D

if dim==2:
    for k in range(len(x1_values)):
        dist_point=[]   # All distances from one point
        for i in range(len(x1_values)):
            dist=math.sqrt((x1_values[k]-x1_values[i])**2+(x2_values[k]-x2_values[i])**2)
            
            all_list_dist.append(dist)
            dist_point.append(dist)
            
        dist_point.remove(0)
        list_mindist.append(min(dist_point))
        
    for k in range(len(x1_values)):       
        for i in range(k,len(x1_values)):
            dist=math.sqrt((x1_values[k]-x1_values[i])**2+(x2_values[k]-x2_values[i])**2)
            list_dist.append(dist)
    list_dist_wo0=[_ for _ in list_dist if _!=0] # every distance between points, without 0 and peers
    s=0
    for i in range(len(list_dist_wo0)):
        s+=list_dist_wo0[i]
    print('The average distance between points is',s/len(list_dist_wo0))
             
    s=0
    for i in range(len(list_mindist)):
        s+=list_mindist[i]  
    print('The average of the distances from the nearest point is',s/len(list_mindist))   
    print('The greatest distance is',max(all_list_dist))



####

# Plot 3D
               
if dim==3:
    for k in range(len(x1_values)):
        dist_point=[]
        for i in range(len(x1_values)):
            dist=math.sqrt((x1_values[k]-x1_values[i])**2+(x2_values[k]-x2_values[i])**2+(x3_values[k]-x3_values[i])**2)
            
            all_list_dist.append(dist)
            dist_point.append(dist)
            
        dist_point.remove(0)
        list_mindist.append(min(dist_point))
        
    for k in range(len(x1_values)):       
        for i in range(k,len(x1_values)):
            dist=math.sqrt((x1_values[k]-x1_values[i])**2+(x2_values[k]-x2_values[i])**2+(x3_values[k]-x3_values[i])**2)
            list_dist.append(dist)
    list_dist_wo0=[_ for _ in list_dist if _!=0] 
    s=0
    for i in range(len(list_dist_wo0)):
        s+=list_dist_wo0[i]
    
    print()
    print('Euclidian Distance (in mm):')
    print()
    av_dist=s/len(list_dist_wo0)
    print('-  The average distance between points is',round(av_dist,3))
             
    s=0
    for i in range(len(list_mindist)):
        s+=list_mindist[i]  
    print('-  The average of the distances from the nearest point is',s/len(list_mindist))   
    print('-  The greatest distance is',max(all_list_dist))

    s=0
    for k in range(len(dist_point)):
        s+=dist_point[k]
        
    total_distance_path=s  
    print('-  The total distance of the path is',round(total_distance_path,3))
    print()




###########################      Volume calculation      ###########################

# if len(bounds)==3:
#     print('Volume Calculation (in mm^3):')
    
#     Vp=4/3*math.pi*r**3    # Particle volume
#     Vc=dime_XY**2*dime_Z   # Cube volume 
    
#     print()
#     print('Dimension :',dime_XY,'x',dime_XY,'x',dime_Z)
#     print( '-  Vp =',round(Vp,5),'and Vc =',round(Vc,5))
    
#     V_path=math.pi*r**2*total_distance_path
    
#     print('-  The volume of the path is about V_path=',round(V_path,3))
    
#     rate=(V_path/Vc)*100
#     print('-  An approximation of V_path / Vc is',round(r,1),'% with',num_samples,'samples')
            
            
            
            
