import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import signal
from scipy.signal import savgol_filter
from scipy import interpolate

##############################################################################    
def read_counts(t,amp_id,sampling_time):
    """Function that read the counts for a single detector
    
    Input:
        - t : Total time of RPT sampling [min]
        - amp_id : Amplifier id which shows in which order the RPT PC
        write the data in the .txt file [-]
        - sampling_time : Sampling time of the experiment [ms]
    Output:
        - count : Counts read by the detector [-] 
        
    """
    
    #open the file
    count_reading= np.loadtxt("./counts.txt")
    
    #write the counts in a vector
    all_counts=[]
    for i in range(len(count_reading)):
        all_counts.append(count_reading[i])
    
    count=[]        
    time=int((t*60*1000)/sampling_time)
    for i in range (0,time):
        count.append(all_counts[7800+(i*26)+(amp_id-1)]) 
    return count
############################################################################## 


##############################################################################         
def write_count_all_det(amps_id_vector,t,sampling_time):
    """Function that writes the counts for the experiment in a .txt file
    
    Input:
        - amps_id_vector : Id number of amplifiers involved in the experiment[-]
        - t : Total time of RPT sampling [min]
        - sampling_time : Sampling time of the experiment [ms]
    Output:
        - counts_all.txt : A .txt file that contains the counts for all the 
        detectors used [-]
        
    """
    
    counts_vector=[]
    for i in range(len(amps_id_vector)):
        count=read_counts(t,amps_id_vector[i],sampling_time)
        counts_vector.append(count)
    
    vector_length = len(counts_vector[0])
    with open("counts_all.txt", "w") as file:
        for i in range(vector_length):
            row = "\t".join(str(vector[i]) for vector in counts_vector)
            file.write(row + "\n")           
############################################################################## 
  
      
##############################################################################
def write_count_handshake_detector(t,amp_id,sampling_time):
    """Function write a .txt file for the handshake
    
    Input:
        - t : Total time of RPT sampling [min]
        - amp_id : Id number of amplifier used for the handshake [-]
        - sampling_time : Sampling time of the experiment [ms]
    Output:
        - handshake_det.txt : A .txt file that contains the counts from the 
        amplifier used for the handshake [-]
        
    """ 
   
    count=read_counts(t,amp_id,sampling_time)
    with open("handshake_det.txt", "w") as file:
        for num in count:
            file.write(str(num) + "\n")
##############################################################################


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#To achieve synchronization, we employ a single detector where we move 
#the particle towards it, perpendicular to the detector's face,
# and subsequently retract it.
 
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


##############################################################################
def read_count():
    """Function that reads the count number of the detector that we used for 
    the synchronization
    
    Input:

    Output:
        - count : Counts for the handshake [-] 
        
    """ 
    
    #read the count number of the detector that we used for synchronization
    count_reading = np.loadtxt("./handshake_det.txt")   
    count=[]
    
    for i in range(len(count_reading)):
        count.append(count_reading[i])
    
    return count
##############################################################################    
    

##############################################################################
def cross_correlation(time,sampling_time,index):
    """Function that calculates the lag time between the robot and the \
    detectors
    
    Input:
        - time : The minimum duration of the particle round movement in front 
        of the detector [s]
        - sampling_time : The interval of time to record the count [ms]
        - index : The number of data points recorded by the robot for 
        TCP position during a single back-and-forth movement to achieve 
        synchronization [-] 
    Output:
        - lag : Amount of calculated time for the lag between the RPT start
        point and the robot start point [s] 
        
    """ 
    
    count=read_count()   
    count_first_cycle=[]     
    for i in range (0,int(time/sampling_time)):
        count_first_cycle.append(count[i])
        
    
    #normalizing the counts to visualize it better with the position data
    normalized_count = (count_first_cycle-np.min(count_first_cycle))/(np.max(count_first_cycle)-np.min(count_first_cycle)) 
    #read the data of TCP position from the robot
    df = np.loadtxt('x_y_robot_position.txt', delimiter=',')
    pos_x=[]
    
    for i in range(index):
        pos_x.append(df[i, 0]-df[0,0])

    normalized_pos = (pos_x - np.min(pos_x)) / (np.max(pos_x) - np.min(pos_x))  

    #performing the cross correlation
    count_np = np.array(normalized_count)
    pos_x_np= np.array(normalized_pos)
    correlation = np.correlate(count_np, pos_x_np, mode='full')  
    lags = signal.correlation_lags(count_np.size, pos_x_np.size, mode="same")
    lag = lags[np.argmax(correlation)]
    print(lag)
    return lag

##############################################################################   


##############################################################################
def visualisation(time,sampling_time,index,lag):
    """Function that produces of plot the counts and the robot position during the part of the 
    experiment for the handshake
    
    Input:
        - time : The minimum duration of the particle round movement in front 
        of the detector [s]
        - sampling_time : The interval of time to record the count [ms]
        - index : The number of data points recorded by the robot for 
        TCP position during a single back-and-forth movement to achieve 
        synchronization [-]
        - lag : Amount of calculated time for the lag between the RPT start
        point and the robot start point [s] 
        
    Output:
        - Plot of the counts and the robot position for a visualisation of the handshake [-]   
        
    """ 
    
    count=read_count()   
    count_first_cycle=[]     
    for i in range (0,int(time/sampling_time)):
        count_first_cycle.append(count[i])
        
    
    #normalizing the counts to visualize 
    normalized_count = (count_first_cycle - np.min(count_first_cycle)) / (np.max(count_first_cycle) - np.min(count_first_cycle)) 
     
    time_x_axis=[]        
    for i in range (0,int(time/sampling_time)):
        time_x_axis.append(i)
        
    plot=matplotlib.pyplot.scatter(time_x_axis,normalized_count,color='black',
                                   s=0.5,label='Photon count') 
    
    
    df = pd.read_csv("x_y_robot_position.txt", sep=",")
    pos_x=[]
    for i in range(index):
        pos_x.append(df.iat[i, 0]-df.iat[0,0])

    normalized_pos = (pos_x - np.min(pos_x)) / (np.max(pos_x) - np.min(pos_x))
    time= np.loadtxt("./time.txt")
    time_vector=[]
            
    #(index*2) is becuase we record the time before and after 
    #each position recording to do average for the sake of precision

    for i in range (0,(index*2)):      
        if (i % 2) == 0:
            average = (((time[i]+time[i+1])/2)+lag)
            time_vector.append((average*100))
    
    plot=matplotlib.pyplot.scatter(time_vector,normalized_pos,color='red',
                                   label='Position_x',s=0.05)      
    plt.xlabel(r'Sampling time',size=18)
    plt.savefig("one cycle",dpi=500)
##############################################################################


##############################################################################
def read_file(file_name):
    """Function that reads the data in a text file and convert it into an array
    
    Input:
        - file_name : The name of file which contains the recorded counts 
        from all the detectors
    Output:
        - vectors : Array containing the data in the text file [-]    
        
    """ 
    
    vectors = []
    with open(file_name, 'r') as file:
        lines = file.readlines()
        num_columns = len(lines[0].split())


        for i in range(num_columns):
            vectors.append([])

        for line in lines:
            values = line.split()
            for i, value in enumerate(values):
                vectors[i].append((value))
    
    return vectors
##############################################################################


##############################################################################    
def write_synchronized_data(file_name,time,sampling_time,lag):
    """Function that produces a .txt file which contains the shifted counts
    
    Input:
        - file_name : The name of file which contains the recorded counts 
        from all the detectors [-]
        - time : Total time of the experiment [min]
        - sampling_time : The interval of time to record the count [ms]
        - lag : Amount of calculated time for the lag between the RPT start
        point and the robot start point [s] 
        
    Output:
        - shifted_counts_all.txt : A .txt file that contains the shifted counts
        
    """ 
    
    counts_vector=read_file(file_name)

    last_sampling_time= int ((time*60*1000)/sampling_time)
    print(last_sampling_time)
    useless_data=int((lag*1000)/sampling_time)
    shifted_count=[]
    for i in range(len(counts_vector)):
        new_count=[]
        for j in range(useless_data,last_sampling_time):
            new_count.append(counts_vector[i][j])
        shifted_count.append(new_count)
        
    
    vector_length = len(shifted_count[0])
    with open("shifted_counts_all.txt", "w") as file:
        for i in range(vector_length):
            row = "\t".join(str(vector[i]) for vector in shifted_count)
            file.write(row + "\n")                
##############################################################################


##############################################################################
def Savitzky_Golay(file_name,window_length,polyorder):
    """Function that writes the filtered counts for denoising in a .txt file
    
    Input:
        - file_name : The name of file which contains the recorded counts 
        from all the detectors [-]
        - window_length : The window length for the denoising [-]
        - polyorder : Order of the polynomial function [-]
        
    Output:
        - denoised_counts_all.txt : A .txt file that contains the denoised 
        counts [-]
        
    """ 
    
    denoised_count_vector=[]
    counts_vector=read_file(file_name)
    for i in range (len(counts_vector)):
        denoised_count= savgol_filter(counts_vector[i], 
                                      window_length, polyorder)
        denoised_count_vector.append(denoised_count)
        
    
    vector_length = len(denoised_count_vector[0])
    with open("denoised_counts_all.txt", "w") as file:
        for i in range(vector_length):
            row = "\t".join(str(vector[i]) for vector in denoised_count_vector)
            file.write(row + "\n")  
##############################################################################


##############################################################################
def rpt_timing(file_name):
    """Function that reads the data in a text file and convert it into an array
    
    Input:
        - file_name : The name of file which contains the recorded counts 
        from all the detectors
    Output:
        - vectors : Array containing the data in the text file [-]    
        
    """ 
    
    counts_vector=read_file(file_name)
    
    rpt_time=[]
    for i in range(0,len(counts_vector[0])):
        rpt_time.append(i*10)
        
    return rpt_time

############################################################################## 


############################################################################## 
def robot_timing():
    
    time_robot= np.loadtxt("./time.txt")
    time=[]
    time_vector=[]
    for i in range(len(time_robot)):
        time.append(time_robot[i])
        
    for i in range (0,len(time)-1):      
        if (i % 2) == 0:
            average = (((time[i]+time[i+1])/2))
            time_vector.append((average*1000))

    
    return time_vector
############################################################################## 


############################################################################## 
def interpolation(file_name):
    
    counts_vector=read_file(file_name)
    x_time=rpt_timing(file_name)
    robot_time=robot_timing()
    interpolaited_counts_vector=[]
    for i in range(len(counts_vector)):
        f = interpolate.interp1d(x_time, counts_vector[i])
        y_new = f(robot_time)
        interpolaited_counts_vector.append(y_new)
        
        vector_length = len(interpolaited_counts_vector[0])
        with open("interpolaited_counts_all_dataset3D1.txt", "w") as file:
            for i in range(vector_length):
                row = "\t".join(str(vector[i]) for vector in interpolaited_counts_vector)
                file.write(row + "\n")  
############################################################################## 


############################################################################## 
def cutoff(file_name):
    
    counts_vector=read_file(file_name)
    for i in range(len(counts_vector)):
    	for j in range(6):
    		if counts_vector[i,j]<50:
    			counts_vector[i,j]=0
    with open("cutoff_counts_all_dataset3D1.txt","w") as file:
    	for i in range(len(counts_vector)):
                row = "\t".join(str(vector[i]) for vector in counts_vector)
                file.write(row + "\n") 
    		
	
############################################################################## 
