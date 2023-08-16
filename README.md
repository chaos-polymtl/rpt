# Radioactive Particle Tracking (RPT)

This repository includes all the data, script and codes that were used for reconstructing the position of a radioactive particle using Artificial Neural Network (ANN).\
It is divided into two folders. The folder `Raw_Data_Post_Processing` contains the files that used for post-processing the raw data from the experiments. The second folder, called `ANN_Recostruction`, contains the files that used to train the ANN and also the ANN reconstruction codes.

# Post-processing raw experimental data
The raw data obtained from the experiment requires post-processing before analysing. To facilitate data analysis, a post-processing code has been developed, consisting of five main parts:
1. `Cleaning the raw data`
   
Initially, the code `PostprocessingDatafile_main` reads the original experiment file, `Data.txt`, which contains measurements from all 26 amplifiers. The counts recorded by each amplifier, whether they have been used during the experiment or not, will be extracted and written to another text file called `counts.txt`.

2. `Organizing the counts`
   
The program generates a `.txt` file named `counts_all.txt`, containing all the counts from the amplifiers involved in the experiment. Additionally, the program generates another `.txt` file named `handshake_det.txt`, which includes the counts from the detector used for the handshake to synchronize the clocks.

3. `Synchronization of the robot clock and the RPT clock`
   
To synchronize the RPT clock with the robot clock, the program implements the cross-correlation technique to calculate the lag between them. Cross-correlation involves comparing the count data from the detectors with the position data from the robot during the handshake phase.

4. `Denoising the data`

Due to the stochastic nature of the radioactive data, before using the data to train the ANN, it must be denoised. To achieve this, the program employs the Savitzky-Golay method.
 
5. `Interpolating the data`

In the final step, to address the mismatched sampling times between the robot and the RPT system, we perform data interpolation. The counts from the amplifiers are taken at a consistent sampling interval of 10 milliseconds, while the robot samples its position at seemingly irregular intervals. To rectify this disparity, interpolation is used to synchronize the count data from the amplifiers with the corresponding timestamps of the robot's position samples. This process allows for the estimation of count values at the specific timestamps of the robot's position measurements.




# ANN reconstruction
