import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from pickle import dump, load
from matplotlib import rc
from matplotlib import rcParams
from keras import backend as K
import matplotlib.pyplot as plt
from matplotlib import pyplot
######################################################################
rcParams['font.size'] = 15
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
################################Global variables #####################
# Number of training points
number_of_points_training = 841000
number_of_points_RPT_experiment = 841000
number_of_detectors = 9
# x_0, y_0 and z_0 are the zero of the robot, start point of the handshake
x_0 = 543.548
y_0 = 174.051
z_0 = 966.329
######################################################################


class DataPreprocess:
    def __init__(self, number_of_points_training, number_of_detectors, x_0, y_0, z_0):
        self.number_of_points_training = number_of_points_training
        self.number_of_detectors = number_of_detectors
        self.x_0 = x_0
        self.y_0 = y_0
        self.z_0 = z_0

    def count_position_data(self):

        # Feed file content
        feed = np.zeros([self.number_of_points_training, self.number_of_detectors + 3])
        # File for the number of counts
        filename_counts = 'interpolated_counts.txt'
        data_counts = np.loadtxt(filename_counts, delimiter='\t')

        for i in range(self.number_of_detectors):
            feed[:, i] = data_counts[0:self.number_of_points_training, i]

        filename_pos = 'robot_position.txt'
        data_pos = np.loadtxt(filename_pos, delimiter=',')

        feed[:, self.number_of_detectors] = (data_pos[0:self.number_of_points_training, 0] - self.x_0) / 1000
        feed[:, self.number_of_detectors + 1] = (data_pos[0:self.number_of_points_training, 1] + self.y_0) / 1000
        feed[:, self.number_of_detectors + 2] = (data_pos[0:self.number_of_points_training, 2] - self.z_0) / 1000

        # Write the counts and position on a text file
        with open("feed.txt", "w") as file_output:
            file_output.truncate(0)
            for row in feed:
                line = '\t'.join([str(item) for item in row])
                file_output.write(f'{line}\n')

        return data_counts, data_pos, feed

    def train_test_data(self):
        # Read the dataset
        self.count_position_data()
        pd_dat = pd.read_csv('feed.txt', delimiter='\t')
        dataset = pd_dat.values

        # Split the dataset into features and target variables
        X_raw = dataset[:, :self.number_of_detectors]
        Y_raw = dataset[:, self.number_of_detectors:]

        # Split the dataset into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X_raw, Y_raw, test_size=0.2)

        # Standardize the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Transpose the scaled features for easy access
        X_train_transposed = np.transpose(X_train_scaled)

        # Dynamically create variables for train inputs
        inputs_train = []
        for i in range(X_train_transposed.shape[0]):
            var_name = f"count{i + 1}_train"
            globals()[var_name] = X_train_transposed[i]
            inputs_train.append(globals()[var_name])

        # Do the same for test inputs
        X_test_transposed = np.transpose(X_test_scaled)
        inputs_test = []
        for i in range(X_test_transposed.shape[0]):
            var_name = f"count{i + 1}_test"
            globals()[var_name] = X_test_transposed[i]
            inputs_test.append(globals()[var_name])

        # Split the targets into individual components
        x_train, y_train, z_train = Y_train.T  # Using .T to transpose for easier unpacking
        x_test, y_test, z_test = Y_test.T

        # Organize the outputs for train and test
        outputs_train = [x_train, y_train, z_train]
        outputs_test = [x_test, y_test, z_test]

        return scaler, inputs_train, outputs_train, inputs_test, outputs_test

    def spiral_data(self):

        data_counts, _ , _= self.count_position_data()
        pred = data_counts[-10000:-1, :self.number_of_detectors]

        return pred

    def handshake_line_data(self):

        data_counts, data_pos, _ = self.count_position_data()
        line = data_counts[1:1500, :self.number_of_detectors]

        return line


class ANNModel:
    def __init__(self, number_of_detectors):
        self.number_of_detectors = number_of_detectors

    def build_model(self):
        inputs = []
        for i in range(1, self.number_of_detectors + 1):
            inputs.append(keras.Input(shape=(1,), name=f"Count{i}"))
        x = layers.concatenate(inputs)

        # Define the hidden layers
        hidden1 = layers.Dense(256, activation='tanh')(x)
        hidden2 = layers.Dense(128, activation='tanh')(hidden1)
        hidden3 = layers.Dense(64, activation='tanh')(hidden2)
        hidden4 = layers.Dense(32, activation='tanh')(hidden3)
        hidden5 = layers.Dense(16, activation='tanh')(hidden4)

        X = layers.Dense(1, activation='linear', name="xx")(hidden5)
        Y = layers.Dense(1, activation='linear', name="yy")(hidden5)
        Z = layers.Dense(1, activation='linear', name="zz")(hidden5)

        model = keras.Model(inputs=inputs, outputs=[X, Y, Z])

        return model

    def run_model(self):
        data_preprocessor = DataPreprocess(number_of_points_training, number_of_detectors, x_0, y_0, z_0)
        scaler, inputs_train, outputs_train, inputs_test, outputs_test = data_preprocessor.train_test_data()

        model = ANNModel(self.number_of_detectors).build_model()
        model.compile(
            optimizer='Adam',
            loss=['mse', 'mse', 'mse'],
            loss_weights=[1.0, 1.0, 1.0],
            metrics=['MAE', 'mean_absolute_percentage_error']
        )
        # Define the learning rate
        LR = 0.00001
        keras.backend.set_epsilon(1)
        K.set_value(model.optimizer.learning_rate, LR)

        history = model.fit(inputs_train, outputs_train,
                            validation_split=0.2,
                            epochs=6000,
                            batch_size=100000,
                            )

        dump(model, open('model.pkl', 'wb'))
        dump(scaler, open('scaler.pkl', 'wb'))
        result = model.evaluate(inputs_test, outputs_test, verbose=2)

        pyplot.subplot(211)
        plt.semilogy(history.history['loss'], label='train')
        plt.semilogy(history.history['val_loss'], label='validation')
        plt.xscale("log")
        pyplot.legend()
        plt.savefig('training_validation_loss.png')  # Save the plot before displaying it

        print(result)


class PostProcessSpiral:
    def __init__(self, number_of_points_training, number_of_detectors, x_0, y_0, z_0):
        self.number_of_points_training = number_of_points_training
        self.number_of_detectors = number_of_detectors
        self.x_0 = x_0
        self.y_0 = y_0
        self.z_0 = z_0

    def prediction_spiral(self):

        model = load(open('model.pkl', 'rb'))
        scaler = load(open('scaler.pkl', 'rb'))

        data_preprocessor = DataPreprocess(number_of_points_training, number_of_detectors, x_0, y_0, z_0)
        pred = data_preprocessor.spiral_data()
        X_pre = pred
        X_pre_scaled = scaler.transform(X_pre)
        X_pre_transposed = np.transpose(X_pre_scaled)

        inputs_pre = []
        for i in range(X_pre_transposed.shape[0]):
            var_name = f"count{i + 1}_pre"
            globals()[var_name] = X_pre_transposed[i]
            inputs_pre.append(globals()[var_name])

        predicted_position = model.predict(inputs_pre)

        return predicted_position

    def error_spiral(self, x_0, y_0, z_0):
        data_preprocessor = DataPreprocess(number_of_points_training, number_of_detectors, x_0, y_0, z_0)
        _, data_pos, _ = data_preprocessor.count_position_data()

        # Real position of the robot during the spiral
        x_real = (data_pos[-10000:-1, 0] - x_0) / 10
        y_real = (data_pos[-10000:-1, 1] + y_0) / 10
        z_real = (data_pos[-10000:-1, 2] - z_0) / 10

        # Convert the data to a NumPy array
        x_pred = np.zeros(len(x_real))
        y_pred = np.zeros(len(y_real))
        z_pred = np.zeros(len(z_real))

        predicted_position = self.prediction_spiral()
        for i in range(len(x_pred)):
            x_pred[i] = predicted_position[0][i][0] * 100
            y_pred[i] = predicted_position[1][i][0] * 100
            z_pred[i] = predicted_position[2][i][0] * 100

        error_x = np.abs(x_pred - x_real)
        MAE_x = error_x.mean() * 10

        error_y = np.abs(y_pred - y_real)
        MAE_y = error_y.mean() * 10

        error_z = np.abs(z_pred - z_real)
        MAE_z = error_z.mean() * 10

        print('MAE_x=', MAE_x, 'mm\nMAE_y =', MAE_y, 'mm\nMAE_z =', MAE_z, 'mm')

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot(x_real, y_real, z_real, label='Robot trajectory', color="black")
        ax.plot(x_pred, y_pred, z_pred, label='Reconstructed trajectory', color="red")
        ax.legend()
        ax.set_xlim(-1, 21)
        ax.set_ylim(-10, 10)
        ax.set_zlim(-20, 1)
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
        ax.set_zlabel('z (cm)')
        plt.savefig('3d_trajectory.png')  # Save the 3D plot
        plt.close(fig)  # Close the figure to free up memory

        # 2d plot x-y
        fig = plt.figure()
        ax = fig.add_subplot()
        ax = plt.figure().add_subplot()
        ax.plot(x_real, y_real, label='Robot trajectory', linewidth=1, color="black")
        ax.plot(x_pred, y_pred, label='Reconstructed trajectory', linewidth=1, color="red")
        ax.grid(axis='both', which='both', color='black', linestyle='-', linewidth=0.2)
        ax.minorticks_on()
        ax.legend()
        ax.set_xlim(-5, 25)
        ax.set_ylim(-15, 15)
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
        plt.savefig('2d_xy_trajectory.png')  # Save the 2D x-y plot
        plt.close(fig)  # Close the figure to free up memory

        # 2d plot x-z
        fig = plt.figure()
        ax = fig.add_subplot()
        ax = plt.figure().add_subplot()
        ax.plot(x_real, z_real, label='Robot trajectory', linewidth=1, color="black")
        ax.plot(x_pred, z_pred, label='Reconstructed trajectory', linewidth=1, color="red")
        ax.grid(axis='both', which='both', color='black', linestyle='-', linewidth=0.2)
        ax.minorticks_on()
        ax.legend()
        ax.set_xlim(-5, 25)
        ax.set_ylim(-25, 5)
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('z (cm)')
        plt.savefig('2d_xz_trajectory.png')  # Save the 2D x-z plot
        plt.close(fig)

        ax.grid(which='both', color='black', linestyle='-', linewidth=0.2)
        ax.minorticks_on()


class FixLag:
    def __init__(self, number_of_points_training, number_of_detectors, x_0, y_0, z_0):
        self.number_of_points_training = number_of_points_training
        self.number_of_detectors = number_of_detectors
        self.x_0 = x_0
        self.y_0 = y_0
        self.z_0 = z_0

    def prediction_handshake(self):

        model = load(open('model.pkl', 'rb'))
        scaler = load(open('scaler.pkl', 'rb'))

        data_preprocessor = DataPreprocess(number_of_points_training, number_of_detectors, x_0, y_0, z_0)
        line = data_preprocessor.handshake_line_data()
        X_pre = line
        X_pre_scaled = scaler.transform(X_pre)
        X_pre_transposed = np.transpose(X_pre_scaled)

        inputs_pre = []
        for i in range(X_pre_transposed.shape[0]):
            var_name = f"count{i + 1}_pre"
            globals()[var_name] = X_pre_transposed[i]
            inputs_pre.append(globals()[var_name])

        predicted_position = model.predict(inputs_pre)

        return predicted_position

    def error_handshake(self):
        predicted = self.prediction_handshake()
        data_preprocessor = DataPreprocess(number_of_points_training, number_of_detectors, x_0, y_0, z_0)
        line = data_preprocessor.handshake_line_data()
        _, _, pos = data_preprocessor.count_position_data()
        error = predicted[0][:] - pos[1:1500, self.number_of_detectors]
        error = error.mean()

        print('\n Error lag x : ', error)




