######################################################################
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
from sklearn.preprocessing import StandardScaler
import re
import csv
from keras import backend as K
from matplotlib import rc
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator
from pickle import load
import math
from matplotlib.tri import Triangulation
from scipy.signal import savgol_filter
import mplcursors
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
nz = 120  # in z direction
min_occurrence = 500
# Savitzky-Golay filter parameters
window_size = 10
order = 2
######################################################################


def predict_position():
    """
    Parameters:
    
    -number_of_points: Number of positions
    
    """
    # import all the counts from the trajectory
    file_path = 'counts_all.txt'
    line_count = sum(1 for _ in open(file_path, 'r'))
    print(line_count)
    # Feed file content
    Feed = np.zeros([number_of_points, 9])

    # File for the number of counts
    filename_counts = 'counts_all.txt'
    data_counts = np.loadtxt(filename_counts, delimiter='\t')

    for i in range(9):
        Feed[:, i] = data_counts[0:number_of_points, i]

        # load the trained model
    model = load(open('model.pkl', 'rb'))
    # load the scaler
    scaler = load(open('scaler.pkl', 'rb'))
    Pred = data_counts[-number_of_points:-1, :9]
    X_pre = Pred
    scaled_X_pre = scaler.transform(X_pre)
    (count1_pre, count2_pre, count3_pre, count4_pre, count5_pre, count6_pre, count7_pre,
     count8_pre, count9_pre) = np.transpose(scaled_X_pre)
    prediction = model.predict([count1_pre, count2_pre, count3_pre, count4_pre, count5_pre, count6_pre,
                                count7_pre, count8_pre, count9_pre])

    # Convert the data to a NumPy array
    predicted_x_pos = np.zeros(number_of_points)
    predicted_y_pos = np.zeros(number_of_points)
    predicted_z_pos = np.zeros(number_of_points)

    for i in range(len(predicted_x_pos) - 1):
        predicted_x_pos[i] = prediction[0][i][0] * 100  # convert to cm
        predicted_y_pos[i] = prediction[1][i][0] * 100
        predicted_z_pos[i] = prediction[2][i][0] * 100

    # Apply Savitzky-Golay filter

    # Polynomial order
    filtered_x = savgol_filter(predicted_x_pos, window_size, order)
    filtered_y = savgol_filter(predicted_y_pos, window_size, order)
    filtered_z = savgol_filter(predicted_z_pos, window_size, order)

    return filtered_x, filtered_y, filtered_z

    


def translation():
    """
    to find the origin
    this part first find the origin of the available data and then shift it
    here we also shift x to have the center of x also 0
    then the new origin would be (0,0) instead of (10,0)ish
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
    calculate the r and z of each point in trajectory
    """

    x_translated, y_translated = translation()

    r_component = np.zeros(number_of_points)
    z_component = np.zeros(number_of_points)

    for i in range(0, number_of_points):
        r_component[i] = np.sqrt(np.square(x_translated[i]) + np.square(y_translated[i]))
        z_component[i] = z_pred[i]

    max_r = np.ceil(r_component.max())
    min_z = -np.ceil(-z_component.min())  # to round up this negative band to a higher negative number
    max_z = np.ceil(z_component.max())

    return r_component, z_component, max_r, min_z, max_z


def calculate_the_midpoints():
    x_translated, y_translated = translation()

    x_midpoint = np.zeros(number_of_points - 1)
    y_midpoint = np.zeros(number_of_points - 1)
    z_midpoint = np.zeros(number_of_points - 1)

    for i in range(0, number_of_points - 1):
        x_midpoint[i] = (x_translated[i + 1] + x_translated[i]) / 2
        y_midpoint[i] = (y_translated[i + 1] + y_translated[i]) / 2
        z_midpoint[i] = (z_pred[i + 1] + z_pred[i]) / 2

    r_midpoint = np.zeros(number_of_points - 1)
    for i in range(0, number_of_points - 1):
        r_midpoint[i] = np.sqrt(np.square(x_midpoint[i]) + np.square(y_midpoint[i]))

    return r_midpoint, z_midpoint


def midpoints_velocity():
    # calculate the v_r and v_z of midpoints

    r, z, r_max, z_min, z_max = xy_to_rz()

    v_r_midpoint = np.zeros(number_of_points - 1)
    v_z_midpoint = np.zeros(number_of_points - 1)

    for i in range(0, number_of_points - 1):
        v_r_midpoint[i] = (r[i + 1] - r[i]) / sampling_time
        v_z_midpoint[i] = (z[i + 1] - z[i]) / sampling_time

    return v_r_midpoint, v_z_midpoint


def mesh():
    r, z, r_max, z_min, z_max = xy_to_rz()

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

    v_r_mid, v_z_mid = midpoints_velocity()
    r_mid, z_mid = calculate_the_midpoints()
    rr, zz = mesh()

    key_cell_index = []
    for i in range(0, nr - 1):
        for j in range(0, nz - 1):
            key_cell_index.append((i, j))

    dict_v_r_component = {}
    dict_v_z_component = {}

    # initialization
    for i in key_cell_index:
        dict_v_r_component[i] = []

    for i in key_cell_index:
        dict_v_z_component[i] = []

    for i in range(0, (number_of_points - 1)):
        point_to_find = (r_mid[i], z_mid[i])
        if r_mid[i] < radius:  # to remove points outside of domain based on r, I did not remove extra z yet.
            key = find_cell_index([rr, zz], point_to_find)  # key output the x_index and y_index
            dict_v_r_component[key].append(v_r_mid[i]/100)
            dict_v_z_component[key].append(v_z_mid[i]/100)

    return dict_v_r_component, dict_v_z_component


def calculate_standard_deviation_of_velocity():

    # calculate the SD of velocity in each cell
    dict_v_r, dict_v_z = assign_velocity_to_cells()
    # v_r
    standard_deviation_v_r = {key: np.std(values) if values else 0 for key, values in dict_v_r.items()}
    Radial_velocity_sd = np.array(list(standard_deviation_v_r.values()))  # to convert cm/sec to m/sec
    # v_z
    standard_deviation_v_z = {key: np.std(values) if values else 0 for key, values in dict_v_z.items()}
    Axial_velocity_sd = np.array(list(standard_deviation_v_z.values()))

    return standard_deviation_v_r, standard_deviation_v_z, Axial_velocity_sd, Radial_velocity_sd


def average_velocity_at_each_cell():

    dict_v_r, dict_v_z = assign_velocity_to_cells()
    # average the v_r at each cell
    average_v_r_component = {key: np.mean(values) if values else 0 for key, values in dict_v_r.items()}
    Radial_velocity_component = np.array(list(average_v_r_component.values()))
    # average the v_z at each cell
    average_v_z_component = {key: np.mean(values) if values else 0 for key, values in dict_v_z.items()}
    Axial_velocity_component = np.array(list(average_v_z_component.values()))

    return average_v_r_component, Radial_velocity_component, average_v_z_component, Axial_velocity_component


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

    dict_v_r, dict_v_z = assign_velocity_to_cells()
    distribution_dict = {}
    for key, vector in dict_v_r.items():
        # Count the total number of objects in the vector
        total_count = len(vector)
        # Store the total count in the new dictionary
        distribution_dict[key] = total_count

    num_occurrence = list(distribution_dict.values())

    return num_occurrence


def plot_occurrence_distribution():

    center_point = write_cell_center()
    num_occurrence = number_of_occurrence_at_each_cell()
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
        num_occurrence = np.delete(num_occurrence, reversed_vector[i])

    # Create a Triangulation
    triang = Triangulation(center_x, center_y)
    # Create a filled contour plot using plt.tripcolor
    plt.tripcolor(triang, num_occurrence, cmap="coolwarm", shading='gouraud')
    cbar = plt.colorbar()
    cbar.set_label('Values')
    plt.gca().set_aspect('equal', adjustable='box')
    # Set labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('occurrence distribution')
    plt.savefig('occurrence distribution.png')  # Save the figure
    plt.show()


def plot_velocity():

    center_point = write_cell_center()
    num_occurrence = number_of_occurrence_at_each_cell()
    average_v_r_component, Radial_velocity, average_v_z_component, Axial_velocity = average_velocity_at_each_cell()
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
        Radial_velocity = np.delete(Radial_velocity, reversed_vector[i])
        num_occurrence = np.delete(num_occurrence, reversed_vector[i])

    for i in range(0, len(num_occurrence)):
        if num_occurrence[i] < min_occurrence:
            Axial_velocity[i] = 0
            Radial_velocity[i] = 0

    # Create a Triangulation
    triang = Triangulation(center_x, center_y)
    # Create a filled contour plot using plt.tripcolor
    plt.tripcolor(triang, Axial_velocity, cmap="coolwarm",
                            shading='gouraud')  # /100 is for converting cm/sec to m/sec
    cbar = plt.colorbar()
    cbar.set_label('Values')
    plt.gca().set_aspect('equal', adjustable='box')
    # Set labels and title
    plt.xlabel('X-axis')

    plt.ylabel('Y-axis')

    plt.title('Axial Velocity Contour')

    plt.savefig('axial_velocity_contour.png', dpi=500)  # Save the figure

    plt.clf()

    plt.tripcolor(triang, Radial_velocity, cmap="coolwarm",
                            shading='gouraud')  # /100 is for converting cm/sec to m/sec

    cbar = plt.colorbar()

    cbar.set_label('Values')

    plt.gca().set_aspect('equal', adjustable='box')

    # Set labels and title

    plt.xlabel('X-axis')

    plt.ylabel('Y-axis')

    plt.title('Radial Velocity Contour')

    plt.savefig('radial_velocity_contour.png', dpi=500)  # Save the figure


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


#####################################################################


x_pred, y_pred, z_pred = predict_position()
plot_occurrence_distribution()
plot_velocity()
plot_sd()
