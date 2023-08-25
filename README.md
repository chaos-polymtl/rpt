# Radioactive Particle Tracking (RPT)

This repository includes all the data, script and codes that were used for reconstructing the position of a radioactive particle using Artificial Neural Network (ANN).\
It is divided into four folders. The folder `Raw_Data_Post_Processing` contains the files that used for post-processing the raw data from the experiments. The second folder, called `ANN_Recostruction`, contains the files that used to train the ANN and also the ANN reconstruction codes. The third folder, named `Robot_motion`, contains the files that are used for a variety of patterns for the robot's motion within the domain of interest. The fourth folder, named `LHS`, contains the codes for generating sampling points within the domain using the Latin Hypercube Sampling (LHS) method.

# Robot motion

We employ a robotic system to move a radioactive particle within our area of interest. This motion yields a substantial dataset comprising precise particle positions and the corresponding photon counts captured by detectors strategically positioned around the domain.

# Latin Hypercube Sampling (LHS)

LHS is a statistical method used to generate a nearly-random sample of parameter values from a multidimensional distribution. We utilize this method to generate points within the domain. The robot moves the particle between these nearly-random points to effectively sample the volume. This folder contains the respective codes for generating the points.

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

The ANN reconstruction folder includes different codes for particle position reconstruction in 1D, 2D, and 3D. The first step is fixing the lag between RPT system and robot properly.

### Fix the lag using ANN

The cross-correlation method employed to address the lag between the RPT system and the robot isn't perfectly precise. Hence, we use ANN to accurately rectify the lag, following the steps outlined below:
1. Identify a line in either the x or y direction along which the robot underwent motion, and determine its corresponding index points in the position file. One approach to accomplish this is through trial and error, achieved by printing various positions from the position file.
2. Train the ANN with the post-processed data set.
3. Reconstruct the line and assess the error (displacement) along its direction. For example, when dealing with a line along the x-axis, evaluate the error in the x-direction; similarly, for a line along the y-axis, assess the error in the y-direction.
4. Calculate the remaining lag by dividing the determined error (in meters) by the velocity of the robot during the traversal of that line (in m/s). This calculation will yield the remaining lag value in seconds.
5. In the post-processing code, incorporate either an addition or subtraction of the remaining lag to the calculated lag value.
6. Repeat steps 2 to 6 until the evaluated error is below $10^{-4}$ meter.

### Position reconstruction using ANN

After fixing the lag in all of the `ANN_Reconstruction` codes, we follow the subsequent steps:
1. We define the number of training points, denoted as `NUM_TP`. This excludes the points that we intend to avoid training the ANN with, for later reconstruction and testing purposes.
2. We fill the `Feed` vector with the post-processed counts and their corresponding positions. With the knowledge of the home position of the robot, we subtract its position from all the training positions, effectively setting the home position as the origin of the experiment.
3. We set up the concatenation layer to provide input to the initial hidden layer of the ANN.
4. Then, we use the `MinMax` method from sklearn to scale the features in the interval of 0 to 1 to avoid any bias during the training.
5. Finaly, we seperate the samples into two sets: training and testing. We use the `train_test_split` function from sklearn to perform the split.
6. The method implemented utilizes the `TensorFlow` and `Keras` libraries to establish the architecture of the ANN. It constructs a deep network based on the method's arguments that define hyperparameters, including the layer count, neuron count, batch size, number of epochs, and activation function. Subsequently, the method compiles the ANN using the training dataset. Within the method, both the model itself and the ANN's training history are returned. This facilitates the monitoring of the loss function's progression over time.

