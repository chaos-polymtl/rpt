# Radioactive Particle Tracking (RPT)

This repository includes all the data, script and codes that were used for reconstructing the position of a radioactive particle using Artificial Neural Network (ANN).\
It is divided into four folders. The folder `Raw_Data_Post_Processing` contains the files that used for post-processing the raw data from the experiments. The second folder, called `ANN_Recostruction`, contains the files that used to train the ANN and also the ANN reconstruction codes. The third folder, named `Robot_motion`, contains the files that are used for a variety of patterns for the robot's motion within the domain of interest. The fourth folder, named `LHS`, contains the codes for generating sampling points within the domain using the Latin Hypercube Sampling (LHS) method.

# Post-processing raw experimental data

The raw data obtained from the experiment at RPT laboratory at polytechnique Montreal requires post-processing before any type of analysing. To facilitate data analysis, a post-processing code has been developed. This code is developed to prepare a data set for ANN position reconstruction consisting of five main parts:
1. `Cleaning the raw data`
   
Initially, the code `PostprocessingDatafile_main` reads the original experiment file, `Data.txt`, which contains measurements from 26 channels. The counts recorded by each amplifier, whether they have been active during the experiment or not, will be extracted and written to another text file called `counts.txt`.

2. `Organizing the counts`
   
The program generates a `.txt` file named `counts_all.txt`, containing the counts from the amplifiers involved in the experiment. Additionally, the program generates another `.txt` file named `handshake_det.txt`, which includes the counts from the solo detector used for the handshake to synchronize the clocks.

3. `Synchronization of the robot clock and the RPT clock`
   
The RPT system and robot are .To synchronize the RPT clock with the robot clock, the program implements the cross-correlation technique to calculate the lag between them. Cross-correlation involves comparing the count data from the detectors with the position data from the robot during the handshake phase.

4. `Denoising the data`

Due to the stochastic nature of the radioactive data, before using the data to train the ANN, it must be denoised. To achieve this, the program employs the Savitzky-Golay filter.
 
5. `Interpolating the data`

In the final step, to address the mismatched sampling times between the robot and the RPT system, we perform data interpolation. The counts from the amplifiers are taken at a consistent sampling interval of 10 milliseconds, while the robot samples its position at seemingly irregular intervals. To rectify this disparity, interpolation is used to synchronize the count data from the amplifiers with the corresponding timestamps of the robot's position samples. This process allows for the estimation of count values at the specific timestamps of the robot's position measurements.


# ANN reconstruction

The ANN reconstruction folder includes different codes for particle position reconstruction in 1D, 2D, and 3D. In any of the `ANN_Reconstruction` codes, we follow the subsequent steps:
1. We define the number of training points, denoted as `NUM_TP`. This excludes the points that we intend to avoid training the ANN with, for later reconstruction and testing purposes.
2. We fill the `Feed` vector with the post-processed counts and their corresponding positions. With the knowledge of the home position of the robot, we subtract its position from all the training positions, effectively setting the home position as the origin of the experiment.
3. We set up the concatenation layer to provide input to the initial hidden layer of the ANN................................................................................. 

