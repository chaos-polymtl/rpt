######################################################################
# Basic libraries
import math
import re
import csv
import numpy as np
import pandas as pd

# TensorFlow and Keras for machine learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K

# Data preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

# Visualization libraries
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams, cm
from matplotlib import pyplot
from matplotlib.ticker import MultipleLocator
import seaborn as sns
import mplcursors
from matplotlib.tri import Triangulation

# Scientific computations
from scipy.signal import savgol_filter
from scipy.interpolate import griddata

# Additional tools
import pyvista as pv
from pickle import load


######################################################################
rcParams['font.size'] = 15
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

################################Global variables #####################
number_of_points = 3600000
radius = 10  # cm
sampling_time = 0.01  # rpt sampling time, millisecond
r_min = 0  # cm
z_min_tank = -20  # cm
z_max_tank = 0  # cm
nr = 70  # in r direction
nz = 140  # in z direction
min_occurrence = 400
# Savitzky-Golay filter parameters
window_size = 31
order = 2
######################################################################


def predict_position():
    """
    Parameters:
    
    -number_of_points: Number of positions
    
    """
    # import all the counts from the trajectory
    file_path = 'rpt_count_stirred_tank.txt'
    line_count = sum(1 for _ in open(file_path, 'r'))
    print(line_count)
    # Feed file content
    Feed = np.zeros([number_of_points, 9])

    # File for the number of counts
    filename_counts = 'counts_all.txt'
    data_counts = np.loadtxt(filename_counts, delimiter='\t')

    for i in range(9):
        Feed[:, i] = data_counts[0:number_of_points, i]

    filtered_data_counts = np.copy(data_counts)
    for i in range(data_counts.shape[1]):
        # Apply the filter to the i-th column
        filtered_data_counts[:, i] = savgol_filter(data_counts[:, i], 31, 2)

    # load the trained model from the robot
    model = load(open('model.pkl', 'rb'))
    # load the scaler
    scaler = load(open('scaler.pkl', 'rb'))
    Pred = filtered_data_counts[-number_of_points:-1, :9]
    X_pre = Pred
    scaled_X_pre = scaler.transform(X_pre)
    (count1_pre, count2_pre, count3_pre, count4_pre, count5_pre, count6_pre, count7_pre,
     count8_pre, count9_pre) = np.transpose(scaled_X_pre)
    prediction = model.predict([count1_pre, count2_pre, count3_pre, count4_pre, count5_pre, count6_pre,
                                count7_pre, count8_pre, count9_pre])

    
    predicted_x_pos = np.zeros(number_of_points)
    predicted_y_pos = np.zeros(number_of_points)
    predicted_z_pos = np.zeros(number_of_points)

    for i in range(len(predicted_x_pos) - 1):
        predicted_x_pos[i] = prediction[0][i][0] * 100  # convert to cm
        predicted_y_pos[i] = prediction[1][i][0] * 100
        predicted_z_pos[i] = prediction[2][i][0] * 100


    # Apply Savitzky-Golay filter

    filtered_x = savgol_filter(predicted_x_pos, window_size, order)
    filtered_y = savgol_filter(predicted_y_pos, window_size, order)
    filtered_z = savgol_filter(predicted_z_pos, window_size, order)

    return filtered_x, filtered_y, filtered_z


def translation():
    """
    First, determine the origin of the available data and then apply a shift. 
    Additionally, shift the x-coordinate so that its center aligns with zero. 
    As a result, the new origin will be (0, 0) instead of (10, 0).
    """

    """
    Parameters:
    
    -number_of_points: Number of positions
    
    -radius: radius of tank (cm)
    """

    x_translation = radius - np.average(x_pred)
    y_translation = np.average(y_pred)

    translated_x_pos = x_pred + x_translation - radius

    if y_translation < 0:
        translated_y_pos = y_pred - y_translation
    else:
        translated_y_pos = y_pred + y_translation

    return translated_x_pos, translated_y_pos


def xy_to_rz():
    """
    Calculate the rr and zz of each point in the trajectory by converting Cartesian coordinates to cylindrical coordinates.
    """

    x_translated, y_translated = translation()

    r_component = np.zeros(number_of_points)
    theta_component = np.zeros(number_of_points)
    z_component = np.zeros(number_of_points)

    for i in range(0, number_of_points):
        r_component[i] = np.sqrt(np.square(x_translated[i]) + np.square(y_translated[i]))
        theta_component[i] = np.arctan2(y_translated[i], x_translated[i])
        z_component[i] = z_pred[i]

    max_r = np.ceil(r_component.max())
    min_z = -np.ceil(-z_component.min())  # to round up this negative band to a higher negative number
    max_z = np.ceil(z_component.max())

    return r_component, z_component, theta_component, max_r, min_z, max_z


def calculate_the_midpoints():
    """
    calculate the mid point between each two recorded data points.
    """
    x_translated, y_translated = translation()

    x_midpoint = np.zeros(number_of_points - 1)
    y_midpoint = np.zeros(number_of_points - 1)
    z_midpoint = np.zeros(number_of_points - 1)

    for i in range(0, number_of_points - 1):
        x_midpoint[i] = (x_translated[i + 1] + x_translated[i]) / 2
        y_midpoint[i] = (y_translated[i + 1] + y_translated[i]) / 2
        z_midpoint[i] = (z_pred[i + 1] + z_pred[i]) / 2

    r_midpoint = np.zeros(number_of_points - 1)
    theta_midpoint = np.zeros(number_of_points - 1)
    for i in range(0, number_of_points - 1):
        r_midpoint[i] = np.sqrt(np.square(x_midpoint[i]) + np.square(y_midpoint[i]))
        theta_midpoint[i] = np.arctan2(y_midpoint[i], x_midpoint[i])

    return r_midpoint, theta_midpoint, z_midpoint


def midpoints_velocity():
    # calculate the v_r and v_z of midpoints
    r_midpoint, theta_midpoint, z_midpoint = calculate_the_midpoints()

    r, z, theta, r_max, z_min, z_max = xy_to_rz()

    v_r_midpoint = np.zeros(number_of_points - 1)
    v_theta_midpoint = np.zeros(number_of_points - 1)
    v_z_midpoint = np.zeros(number_of_points - 1)

    #calculate deltatheta
    delta_theta = np.zeros(number_of_points - 1)
    for i in range (0,number_of_points-1):
        delta_theta[i] = theta[i + 1] - theta[i]
        if ((theta[i + 1] - theta[i])>np.pi):
            delta_theta[i]= (theta[i + 1] - theta[i])- (2. * np.pi)
        elif ((theta[i + 1] - theta[i])<(-1*np.pi)):
            delta_theta[i] = (theta[i + 1] - theta[i])+ (2. * np.pi)

    for i in range(0, number_of_points - 1):
        v_r_midpoint[i] = (r[i + 1] - r[i]) / sampling_time
        #v_theta_midpoint[i] = r_midpoint[i]*(((theta[i + 1] - theta[i] + np.pi) % (2. * np.pi) - np.pi) / sampling_time)
        v_theta_midpoint[i] = (r_midpoint[i]*delta_theta[i]) / sampling_time
        v_z_midpoint[i] = (z[i + 1] - z[i]) / sampling_time

    return v_r_midpoint, v_theta_midpoint, v_z_midpoint


def mesh():
    r, z, theta, r_max, z_min, z_max = xy_to_rz()

    r_grid = np.linspace(0, r_max, nr)
    z_grid = np.linspace(z_min, z_max, nz)
    rr_grid, zz_grid = np.meshgrid(r_grid, z_grid)

    return rr_grid, zz_grid


def find_cell_index(meshgrid, point):
    x_values = meshgrid[0][0, :]
    y_values = meshgrid[1][:, 0]
    found_x_index = np.searchsorted(x_values, point[0], side='right')-1
    found_y_index = np.searchsorted(y_values, point[1], side='right')-1

    return found_x_index, found_y_index


def assign_velocity_to_cells():

    v_r_mid, v_theta_mid, v_z_mid = midpoints_velocity()
    r_mid, theta_mid, z_mid = calculate_the_midpoints()
    rr, zz = mesh()

    key_cell_index = []
    for i in range(0, nr - 1):
        for j in range(0, nz - 1):
            key_cell_index.append((i, j))

    dict_v_r_component = {}
    dict_v_theta_component = {}
    dict_v_z_component = {}

    # initialization
    for i in key_cell_index:
        dict_v_r_component[i] = []

    for i in key_cell_index:
        dict_v_z_component[i] = []

    for i in key_cell_index:
        dict_v_theta_component[i] = []

    for i in range(0, (number_of_points - 1)):
        point_to_find = (r_mid[i], z_mid[i])
        if r_mid[i] < radius:  # to remove points outside of domain based on r, I did not remove extra z yet.
            key = find_cell_index([rr, zz], point_to_find)  # key output the x_index and y_index
            dict_v_r_component[key].append(v_r_mid[i]/100)
            dict_v_theta_component[key].append(v_theta_mid[i] / 100)
            dict_v_z_component[key].append(v_z_mid[i]/100)

    return dict_v_r_component, dict_v_theta_component, dict_v_z_component


def calculate_standard_deviation_of_velocity():

    # calculate the SD of velocity in each cell
    dict_v_r, dict_v_theta, dict_v_z = assign_velocity_to_cells()
    # v_r
    standard_deviation_v_r = {key: np.std(values) if values else 0 for key, values in dict_v_r.items()}
    Radial_velocity_sd = np.array(list(standard_deviation_v_r.values()))  # to convert cm/sec to m/sec
    # v_z
    standard_deviation_v_z = {key: np.std(values) if values else 0 for key, values in dict_v_z.items()}
    Axial_velocity_sd = np.array(list(standard_deviation_v_z.values()))

    return standard_deviation_v_r, standard_deviation_v_z, Axial_velocity_sd, Radial_velocity_sd


def average_velocity_at_each_cell():

    dict_v_r, dict_v_theta, dict_v_z = assign_velocity_to_cells()
    # average the v_r at each cell
    average_v_r_component = {key: np.mean(values) if values else 0 for key, values in dict_v_r.items()}
    Radial_velocity_component = np.array(list(average_v_r_component.values()))
    # average the v_theta at each cell
    average_v_theta_component = {key: np.mean(values) if values else 0 for key, values in dict_v_theta.items()}
    Tangential_velocity_component = np.array(list(average_v_theta_component.values()))
    # average the v_z at each cell
    average_v_z_component = {key: np.mean(values) if values else 0 for key, values in dict_v_z.items()}
    Axial_velocity_component = np.array(list(average_v_z_component.values()))

    return average_v_r_component, Radial_velocity_component, average_v_theta_component, Tangential_velocity_component, average_v_z_component, Axial_velocity_component


def find_cell_center(meshgrid, cell_indices):

    x_values = meshgrid[0][0, :]
    z_values = meshgrid[1][:, 0]

    x_center = 0.5 * (x_values[cell_indices[0]] + x_values[cell_indices[0] + 1])
    z_center = 0.5 * (z_values[cell_indices[1]] + z_values[cell_indices[1] + 1])

    return x_center, z_center


def write_cell_center():
    rr, zz = mesh()
    center_point = []
    for i in range(0, nr - 1):
        for j in range(0, nz - 1):
            center_point.append(find_cell_center((rr, zz), (i, j)))

    return center_point


def number_of_occurrence_at_each_cell():

    dict_v_r, dict_v_theta, dict_v_z = assign_velocity_to_cells()
    distribution_dict = {}
    for key, vector in dict_v_r.items():
        # Count the total number of objects in the vector
        total_count = len(vector)
        # Store the total count in the new dictionary
        distribution_dict[key] = total_count

    num_occurrence = list(distribution_dict.values())

    return num_occurrence


def fluctuating_velocity():

    dict_v_r, dict_v_theta, dict_v_z = assign_velocity_to_cells()

    for key, values in dict_v_r.items():
        # Calculate the average of the values
        average = sum(values) / len(values) if values else 0

        # Replace each value with the difference between the value and the average
        dict_v_r[key] = [value - average for value in values]

    for key, values in dict_v_z.items():
        # Calculate the average of the values
        average = sum(values) / len(values) if values else 0

        # Replace each value with the difference between the value and the average
        dict_v_z[key] = [value - average for value in values]

    return dict_v_r, dict_v_z



def plot_occurrence_distribution():
    center_point = write_cell_center()
    num_occurrence = number_of_occurrence_at_each_cell()
    center_x = []
    center_y = []
    cell_to_remove = []

    for i in range(0, len(center_point)):
        if center_point[i][0] < radius and z_min_tank < center_point[i][1] < z_max_tank:
            center_x.append(center_point[i][0])
            center_y.append(center_point[i][1] + 20)
        else:
            cell_to_remove.append(i)
        
        # Check for special condition and mark those cells with a distinct value
        if 0 < center_point[i][0] < 0.7 and -14.5 < center_point[i][1] < z_max_tank:
            num_occurrence[i] = -1  # Marking special cells with -1
            
        if 0 < center_point[i][0] <1.9 and  -14.5< center_point[i][1] <-10.7 :
            num_occurrence[i] = -1  # Marking special cells with -1

    reversed_vector = cell_to_remove[::-1]

    for i in range(0, len(reversed_vector)):
        num_occurrence = np.delete(num_occurrence, reversed_vector[i])

    # Create a Triangulation
    triang = Triangulation(center_x, center_y)
    
    # Use the 'YlGnBu' colormap from Matplotlib
    cmap = plt.get_cmap('YlGn')
    cmap.set_under('black')  # Color for special cells

    # Create a filled contour plot using plt.tripcolor
    plt.tripcolor(triang, num_occurrence, cmap=cmap, shading='gouraud', vmin=0)
    cbar = plt.colorbar()
    # cbar.set_label('Values')
    
    # Customize x-axis ticks
    plt.xticks([0, 5, 10])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('r (cm)')
    plt.ylabel('z (cm)')
    
    # Save the figure
    plt.savefig('occurrence_distribution.eps', bbox_inches='tight',dpi=600)
    plt.show()
    plt.clf()



def plot_velocity():
    center_point = write_cell_center()
    num_occurrence = number_of_occurrence_at_each_cell()
    average_v_r_component, Radial_velocity, average_v_theta_component, Tangential_velocity, average_v_z_component, Axial_velocity = average_velocity_at_each_cell()
    center_x = []
    center_y = []
    cell_to_remove = []

    for i in range(0, len(center_point)):
        if center_point[i][0] < radius and z_min_tank < center_point[i][1] < z_max_tank:
            center_x.append(center_point[i][0])
            center_y.append(center_point[i][1] + 20)
        else:
            cell_to_remove.append(i)

    reversed_vector = cell_to_remove[::-1]

    for i in range(0, len(reversed_vector)):
        Axial_velocity = np.delete(Axial_velocity, reversed_vector[i])
        Tangential_velocity = np.delete(Tangential_velocity, reversed_vector[i])
        Radial_velocity = np.delete(Radial_velocity, reversed_vector[i])
        num_occurrence = np.delete(num_occurrence, reversed_vector[i])

    for i in range(0, len(num_occurrence)):
        if num_occurrence[i] < min_occurrence:
            Axial_velocity[i] = np.nan
            Tangential_velocity[i] = np.nan
            Radial_velocity[i] = np.nan

    # Create a custom colormap that includes white for NaN values
    cmap = plt.cm.coolwarm
    cmap.set_bad('white', 1.)

    # Create a Triangulation
    triang = Triangulation(center_x, center_y)
    # Create a filled contour plot using plt.tripcolor for Axial velocity
    plt.tripcolor(triang, Axial_velocity, cmap=cmap, shading='gouraud', vmin=-0.2, vmax=0.2)
    cbar = plt.colorbar()
    plt.xticks([0, 5, 10])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('r (cm)')
    plt.ylabel('z (cm)')
    plt.savefig('axial_mean_RPT.eps', dpi=1200, bbox_inches='tight')
    plt.clf()

    # Create a filled contour plot using plt.tripcolor for Radial velocity
    plt.tripcolor(triang, Radial_velocity, cmap=cmap, shading='gouraud', vmin=-0.1, vmax=0.25)
    cbar = plt.colorbar()
    plt.xticks([0, 5, 10])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('r (cm)')
    plt.ylabel('z (cm)')
    plt.savefig('radial_mean_RPT.eps', dpi=1200, bbox_inches='tight')
    plt.clf()

    # Create a filled contour plot using plt.tripcolor for Tangential velocity
    plt.tripcolor(triang, Tangential_velocity, cmap=cmap, shading='gouraud', vmin=-0.83, vmax=0.0)
    cbar = plt.colorbar()
    plt.xticks([0, 5, 10])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('r (cm)')
    plt.ylabel('z (cm)')
    plt.savefig('tangential_mean_RPT.eps', dpi=1200, bbox_inches='tight')
    plt.show()
    plt.clf()

def plot_sd():

    center_point = write_cell_center()
    num_occurrence = number_of_occurrence_at_each_cell()
    standard_deviation_v_r, standard_deviation_v_z, Axial_velocity_sd, Radial_velocity_sd = calculate_standard_deviation_of_velocity()
    center_x = []
    center_y = []
    cell_to_remove = []

    for i in range(0, (len(center_point))):
        if center_point[i][0] < radius and z_min_tank < center_point[i][1] < z_max_tank:
            center_x.append(center_point[i][0])
            center_y.append(center_point[i][1])
        else:
            cell_to_remove.append(i)

    reversed_vector = cell_to_remove[::-1]

    for i in range(0, len(reversed_vector)):
        Axial_velocity_sd = np.delete(Axial_velocity_sd, reversed_vector[i])
        Radial_velocity_sd = np.delete(Radial_velocity_sd, reversed_vector[i])
        num_occurrence = np.delete(num_occurrence, reversed_vector[i])

    for i in range(0, len(num_occurrence)):
        if num_occurrence[i] < min_occurrence:
            Axial_velocity_sd[i] = 0
            Radial_velocity_sd[i] = 0

    # Create a Triangulation
    triang = Triangulation(center_x, center_y)
    # Create a filled contour plot using plt.tripcolor
    contour = plt.tripcolor(triang, Axial_velocity_sd, cmap="coolwarm",
                            shading='gouraud')  # /100 is for converting cm/sec to m/sec

    cbar = plt.colorbar()
    cbar.set_label('Values')
    plt.gca().set_aspect('equal', adjustable='box')
    # Set labels and title
    plt.xlabel('X-axis')

    plt.ylabel('Y-axis')

    plt.title('Axial Velocity standard deviation Contour')

    plt.savefig('axial_velocity_contour_sd.png', dpi=500)  # Save the figure

    plt.clf()

    contour = plt.tripcolor(triang, Radial_velocity_sd, cmap="coolwarm",
                            shading='gouraud')  # /100 is for converting cm/sec to m/sec

    cbar = plt.colorbar()

    cbar.set_label('Values')

    plt.gca().set_aspect('equal', adjustable='box')

    # Set labels and title

    plt.xlabel('X-axis')

    plt.ylabel('Y-axis')

    plt.title('Radial Velocity standard deviation Contour')

    plt.savefig('radial_velocity_contour_sd.png', dpi=500)  # Save the figure

    plt.clf()


def output_velocity():

    # Extracting velocities and write in a and Excel file
    center_point = write_cell_center()
    average_v_r_component, Radial_velocity, average_v_theta_component, Tangential_velocity, average_v_z_component, Axial_velocity = average_velocity_at_each_cell()
    center_x = []
    center_y = []
    cell_to_remove = []

    for i in range(0, (len(center_point))):
        if center_point[i][0] < radius and z_min_tank < center_point[i][1] < z_max_tank:
            center_x.append(center_point[i][0])
            center_y.append(center_point[i][1])
        else:
            cell_to_remove.append(i)

    reversed_vector = cell_to_remove[::-1]

    for i in range(0, len(reversed_vector)):
        Axial_velocity = np.delete(Axial_velocity, reversed_vector[i])
        Tangential_velocity = np.delete(Tangential_velocity, reversed_vector[i])
        Radial_velocity = np.delete(Radial_velocity, reversed_vector[i])

    # Specify the file path
    file_path = "output_velocity_experiment.txt"

    # Open the file in write mode
    with open(file_path, 'w') as file:
        # Write the header line
        file.write("r z radial_velocity Tangential_velocity axial_velocity\n")

        # Iterate over the elements of the vectors
        for i in range(len(center_x)):
            # Write the elements to the file
            file.write(f"{center_x[i]} {center_y[i]} {Radial_velocity[i]} {Tangential_velocity[i]} {Axial_velocity[i]}\n")


def plot_cfd_axial_profile():

    csv_file_path = 'file.csv'
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)
    # Extract the first and third elements of each row and create a vector of vectors
    first_third_vector = [list(row[['Points_1', 'Points_0']]) for index, row in df.iterrows()]

    # Extract the fourth column and create a separate vector
    ax_velocity = list(df['average_velocity_0'])
    az_velocity = list(df['average_velocity_2'])
    rad_velocity = list(df['average_velocity_1'])

    center_x = []
    center_y = []
    axial_velocity = []
    azimuthal_velocity = []
    radial_velocity = []

    for i in range(0, len(ax_velocity)):
        if first_third_vector[i][0] >= 0:
           center_x.append(first_third_vector[i][0]*100)
           center_y.append((first_third_vector[i][1])*100+10)
           axial_velocity.append(ax_velocity[i])
           azimuthal_velocity.append(az_velocity[i])
           radial_velocity.append(rad_velocity[i])

    # Create a Triangulation
    triang = Triangulation(center_x, center_y)
    # Create a filled contour plot using plt.tripcolor
    plt.tripcolor(triang, axial_velocity, cmap="coolwarm",
                            shading='gouraud', vmin=-0.2, vmax=0.2)  # /100 is for converting cm/sec to m/sec
    cbar = plt.colorbar()
    plt.xticks([0, 5, 10])
    plt.gca().set_aspect('equal', adjustable='box')
    # Set labels and title
    plt.xlabel('r (cm)')
    plt.ylabel('z (cm)')

    plt.savefig('axial_mean_cfd.eps', dpi=1200, bbox_inches='tight')  # Save the figure

    plt.clf()

    plt.tripcolor(triang, radial_velocity, cmap="coolwarm",
                            shading='gouraud', vmin=-0.1, vmax=0.25)  # /100 is for converting cm/sec to m/sec
    cbar = plt.colorbar()
    plt.xticks([0, 5, 10])
    plt.gca().set_aspect('equal', adjustable='box')
    # Set labels and title
    plt.xlabel('r (cm)')
    plt.ylabel('z (cm)')

    plt.savefig('radial_mean_cfd.eps', dpi=1200, bbox_inches='tight')  # Save the figure

    plt.clf()

    plt.tripcolor(triang, azimuthal_velocity, cmap="coolwarm",
                            shading='gouraud', vmin=-0.83, vmax=0)  # /100 is for converting cm/sec to m/sec
    cbar = plt.colorbar()
    plt.xticks([0, 5, 10])
    plt.gca().set_aspect('equal', adjustable='box')
    # Set labels and title
    plt.xlabel('r (cm)')
    plt.ylabel('z (cm)')

    plt.savefig('azimuthal_mean_cfd.eps', dpi=1200, bbox_inches='tight')  # Save the figure

    plt.clf()


def plot_vertex():

    center_point = write_cell_center()
    average_v_r_component, Radial_velocity,average_v_theta_component, Tangential_velocity, average_v_z_component, Axial_velocity = average_velocity_at_each_cell()
    num_occurrence = number_of_occurrence_at_each_cell()
    center_x = []
    center_y = []
    cell_to_remove = []
    normalized_radial = []
    normalized_axial = []

    for i in range(0, (len(center_point))):
        if center_point[i][0] < radius and z_min_tank < center_point[i][1] < z_max_tank:
            center_x.append(center_point[i][0])
            center_y.append(center_point[i][1]+20)
        else:
            cell_to_remove.append(i)

    reversed_vector = cell_to_remove[::-1]

    for i in range(0, len(reversed_vector)):
        Axial_velocity = np.delete(Axial_velocity, reversed_vector[i])
        Radial_velocity = np.delete(Radial_velocity, reversed_vector[i])
        num_occurrence = np.delete(num_occurrence, reversed_vector[i])

    for i in range(0, len(num_occurrence)):
        if num_occurrence[i] < min_occurrence:
            Axial_velocity[i] = 0
            Radial_velocity[i] = 0

    magnitude = (Radial_velocity ** 2 + Axial_velocity ** 2) ** 0.5

    maximum = np.max(magnitude)

    # Normalize the vectors to ensure consistent arrow lengths

    for i in range(0, len(magnitude)):
        normalized_radial.append((Radial_velocity[i] / maximum)*2.5)
        normalized_axial.append((Axial_velocity[i] / maximum)*2.5)

    for i in range(0, (len(center_x)//15)):
        plt.quiver(center_x[i*15], center_y[i*15], normalized_radial[i*15], normalized_axial[i*15], scale=20, color='black', width=0.005, headwidth=3)

    # Customize x-axis ticks
    plt.xticks([0, 5, 10])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('r (cm)')
    plt.ylabel('z (cm)')
    plt.savefig('vertices.eps', bbox_inches='tight',dpi=600)  # Save the figure
    plt.clf()
    

def extrapolate_values():

    center_point = write_cell_center()
    uu_vector, ww_vector = normal_stresses()
    num_occurrence = number_of_occurrence_at_each_cell()
    center_x = []
    center_y = []
    cell_to_remove = []
    diff_uu = []
    diff_ww = []
    for i in range(0, (len(center_point))):
        if center_point[i][0] < radius and z_min_tank < center_point[i][1] < z_max_tank:
            center_x.append(center_point[i][0])
            center_y.append(center_point[i][1]+20)
        else:
            cell_to_remove.append(i)

    reversed_vector = cell_to_remove[::-1]
    for i in range(0, len(reversed_vector)):
        uu_vector = np.delete(uu_vector, reversed_vector[i])
        ww_vector = np.delete(ww_vector, reversed_vector[i])
        num_occurrence = np.delete(num_occurrence, reversed_vector[i])
    for i in range(0, len(num_occurrence)):
        if num_occurrence[i] < min_occurrence:
            uu_vector[i] = 0
            ww_vector[i] = 0

    r1 = np.array(center_x)
    z1 = np.array(center_y)

    # Read data from the first file
    file2 = "cfd_stresses.csv"

    # Read data from the second file using csv module
    with open(file2, 'r') as csv_file:
        lines = csv_file.readlines()
        data2 = np.array([list(map(float, line.strip().split(','))) for line in lines])

    r2, z2, interp_val1, interp_val2 = data2[:, 0], data2[:, 1], data2[:, 2], data2[:, 3]

    #conver cfd positions to experimental positions
    r2 = r2 * 100
    z2 = (z2 * 100) + 10

    # Perform 2D interpolation for the values from the second file
    interp_val1 = griddata((r2.flatten(), z2.flatten()), interp_val1.flatten(),
                           (r1.flatten(), z1.flatten()), method='linear', fill_value=np.nan)

    interp_val2 = griddata((r2.flatten(), z2.flatten()), interp_val2.flatten(),
                           (r1.flatten(), z1.flatten()), method='linear', fill_value=np.nan)
    # Calculate differences

    for i in range(0, len(uu_vector)):
        if uu_vector[i] == 0:
            diff_uu.append(0)
        else:
            diff_uu.append(uu_vector[i] - interp_val1[i])

    for i in range(0, len(ww_vector)):
        if ww_vector[i] == 0:
            diff_ww.append(0)
        else:
            diff_ww.append(abs(ww_vector[i] - interp_val2[i]))

    # Create a Triangulation
    triang = Triangulation(center_x, center_y)
    # Create a filled contour plot using plt.tripcolor
    plt.tripcolor(triang, diff_uu, cmap="coolwarm",
                            shading='gouraud', vmin=0, vmax=0.016)  # /100 is for converting cm/sec to m/sec
    cbar = plt.colorbar()
    # Set labels and title
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xticks([0, 5, 10])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('r (cm)')
    plt.ylabel('z (cm)')
    plt.savefig('diff_uu_6_hr.png', bbox_inches='tight')  # Save the figure
    plt.clf()

    plt.tripcolor(triang, diff_ww, cmap="coolwarm",
                            shading='gouraud', vmin=0, vmax=0.016)  # /100 is for converting cm/sec to m/sec
    cbar = plt.colorbar()
    # Set labels and title
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xticks([0, 5, 10])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('r (cm)')
    plt.ylabel('z (cm)')
    plt.savefig('diff_ww_6_hr.png', bbox_inches='tight')  # Save the figure
    plt.clf()



