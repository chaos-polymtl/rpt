import re
import pandas as pd
from Postprocessing2D_Functions import *
import os
from os.path import exists

####################### Clean Main RPT Output  ###############################

#The raw data output from the RPT system need clean up before any manipulation
#Read the RPT raw text file with the correct encoding

#Write the forth coloumn of the raw data (the only data to be extracted) 
#in counts.txt
with open("Data.txt", "r",encoding='latin-1') as input_file:
    with open("counts.txt", "w") as output_file:
        for line in input_file:
            columns = line.split()
            if len(columns) >= 4:
                output_file.write(columns[3] + "\n")

#open the file
with open('counts.txt', 'r') as file:
        text = file.read()

# Replace all superscript 3 characters with an empty string
text = re.sub(r'³', ' ', text)

# Write the modified text 
with open('counts.txt', 'w') as file:
        file.write(text)
    
#open the file
with open('counts.txt', 'r') as file:
        lines = file.readlines()

#Delete the first 3 lines of the clean_count.txt
with open('counts.txt', 'w') as file:
        file.writelines(lines[3:])
        
Position_file = "pos.csv"

if exists(Position_file)==True:
    base = os.path.splitext(Position_file)[0]
    os.rename(Position_file, base + ".txt")
    os.rename('pos.txt', 'x_y_robot_position.txt')        
        
        
with open('x_y_robot_position.txt','r') as file:
        text = file.read()
text = text.replace("[", "")
text = text.replace("]", "")
text = text.replace('"', "")


# Write the modified text 
with open('x_y_robot_position.txt', 'w') as file:
        file.write(text)

Time_file = "time.csv"
if exists(Time_file)==True:
    base = os.path.splitext(Time_file)[0]
    os.rename(Time_file, base + ".txt")    
##############################################################################

 
####################### Count Organization ###################################

amps_id_vector=[6,8,17,20] # ID of the amplifiers used
amps_id_handshake = 8 # ID of detector used for the handshake
t=256 # Total time in minutes
sampling_time=10 # Sampling time in milliseconds
     
write_count_all_det(amps_id_vector,t, sampling_time)   
write_count_handshake_detector(t, amps_id_handshake, sampling_time)
##############################################################################


######################## Syncronization ######################################
time=60000
index=3459

cross_correlation(time,sampling_time,index)

lag=4.54
visualisation(time,sampling_time,index,lag)
##############################################################################
 

######################## Cut Data ############################################
file_name="counts_all.txt"
write_synchronized_data(file_name,t,sampling_time,lag)
##############################################################################


######################## Denoising ###########################################
window_length=301
polyorder=1
Savitzky_Golay("shifted_counts_all.txt",window_length,polyorder)
##############################################################################


######################## Interpolation #######################################
interpolation("denoised_counts_all.txt")
##############################################################################

plt.show()
