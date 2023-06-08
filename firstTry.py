import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Conv1DTranspose
from tensorflow import keras
import pandas as pd
from obspy.io import reftek
from obspy import read, Trace
from obspy import Stream
from obspy.signal.filter import highpass
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
import glob
import sklearn
from sklearn.utils import shuffle

def load_data(file_paths):
    data = []
    for file_path in file_paths:
        st = read(file_path, format="REFTEK130")
        if not glob.glob(file_path):
            print(f"File not found: {file_path}")
            continue
        st[3:].resample(40.0)
        st[3:].filter("highpass", freq=0.5)
        data.append(st[3:])
    return data


def load_data_Z(file_paths):
    dataZ = []
    for file_path in file_paths:
        st = read(file_path, format="REFTEK130")
        if not glob.glob(file_path):
            print(f"File not found: {file_path}")
            continue
        st[0].resample(40.0)
        st[0].filter("highpass", freq=0.5)
        dataZ.append(st[0])
    return dataZ

def load_data_E(file_paths):
    dataE = []
    for file_path in file_paths:
        st = read(file_path, format="REFTEK130")
        if not glob.glob(file_path):
            print(f"File not found: {file_path}")
            continue
        st[1].resample(40.0)
        st[1].filter("highpass", freq=0.5)
        dataE.append(st[1])
    return dataE

def load_data_N(file_paths):
    dataN = []
    for file_path in file_paths:
        st = read(file_path, format="REFTEK130")
        if not glob.glob(file_path):
            print(f"File not found: {file_path}")
            continue
        st[2].resample(40.0)
        st[2].filter("highpass", freq=0.5)
        dataN.append(st[2])
    return dataN

excel_file = pd.read_excel("/home/kalina/software/ice/Events_90-130_Valio.xlsx")

data_dir = "/home/kalina/software/ice/2020/"  
file_paths = []
file_label_arrays = []

excel_file["Date"] = pd.to_datetime(excel_file["Date"], format="%d/%m/%Y")

for index, row in excel_file.iterrows():
    event_date = row["Date"]
    start_time = row["StartTime"]
    event_type = row["EventType"]
    day_number = event_date.timetuple().tm_yday
    day_dir = f"2020{day_number:03d}/9906/1"
    start_time_hh=start_time.hour
    if (start_time_hh<10):
        file_path = os.path.join(data_dir, day_dir, f"0{start_time_hh}0000000_0036EE80")
    else:
        file_path = os.path.join(data_dir, day_dir, f"{start_time_hh}0000000_0036EE80")
    if os.path.exists(file_path):
        if file_path not in file_paths:
            file_paths.append(file_path)
            label_array = np.zeros(3600*40)
            file_label_arrays.append(label_array)
        else:
            label_array = file_label_arrays[file_paths.index(file_path)]
        event_seconds = int((start_time.minute * 60 + start_time.second)*40)
        label_array[event_seconds] = 100 if event_type == "led" else 100

for file_path, label_array in zip(file_paths, file_label_arrays):
    day_number = int(file_path.split("/")[6][-3:])
    day_date = datetime.datetime.strptime(f"2020{day_number:03d}", "%Y%j").date()
    start_time_hh = file_path.split("/")[9].split("_")[0][:2]
    event_count = np.count_nonzero(label_array)
    print(f"File: {file_path}, Day: {day_date}, Start hour: {start_time_hh}h, Events: {event_count}")

#dataZ = load_data_Z(file_paths)
#dataN = load_data_N(file_paths)
#dataE = load_data_E(file_paths)

data = load_data(file_paths)

def plot_file(data, label_array):
    fig, ax1 = plt.subplots()
    ax1.plot(data)
    ax2 = ax1.twinx()
    ax2.plot(label_array,color='red')
    ax2.set_ylim([0, 2])
    #plt.ylabel("Amplitude")
    #plt.title("Seismic Data with Event Markers")
    plt.show()

def plot_file_2(data, predicted_labels, original_labels):
    fig, ax1 = plt.subplots()
    ax1.plot(data)
    ax2 = ax1.twinx()
    ax2.plot(predicted_labels, color='red', label='Predicted')
    ax2.plot(original_labels, color='blue', label='Original')
    ax2.set_ylim([0, 101])
    ax2.legend(loc='upper right')
    plt.show()

#for i in range(len(file_paths)):
 #   plot_file(data[i], file_label_arrays[i])

print("Number of files: ",len(file_paths))
print("Number of label arrays: ",len(file_label_arrays))

data = np.stack(data)
#dataZ = np.stack(dataZ)
#dataN = np.stack(dataN)
#dataE = np.stack(dataE)
#data = np.vstack((dataZ,dataN,dataE))
file_label_arrays = np.stack(file_label_arrays)
#file_label_arrays = np.vstack((file_label_arrays_1,file_label_arrays_1,file_label_arrays_1))
print("Data initial shape: ",data.shape)
print("Labels initial shape: ",file_label_arrays.shape)
data,file_label_arrays=shuffle(data,file_label_arrays)
nevents,nsamples = data.shape
data = data.reshape(nevents,nsamples,3)
file_label_arrays = file_label_arrays.reshape(nevents,nsamples,1)
print (data.shape)



model = Sequential(
    [
        keras.layers.InputLayer(input_shape=(nsamples,3)),
        keras.layers.Conv1D(filters=32, kernel_size=24, padding="same", strides=1, activation="relu",kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.Dropout(rate=0.3),
        keras.layers.Conv1D(filters=16, kernel_size=12, padding="same", strides=1, activation="relu",kernel_regularizer=keras.regularizers.l2(0.01)),

        keras.layers.Conv1D(filters=8, kernel_size=6, padding="same", strides=1, activation="relu",kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.Conv1DTranspose(filters=8, kernel_size=6, padding="same", strides=1, activation="relu",kernel_regularizer=keras.regularizers.l2(0.01)),

        keras.layers.Conv1DTranspose(filters=16, kernel_size=12, padding="same", strides=1, activation="relu",kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.Dropout(rate=0.3),
        keras.layers.Conv1DTranspose(filters=32, kernel_size=24, padding="same", strides=1, activation="relu",kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.Conv1DTranspose(filters=1, kernel_size=24, padding="same"),
    ]
)

model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.00001))
model.summary()

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


history = model.fit(
    data,
    file_label_arrays,
    epochs=50,
    batch_size=32,
    validation_split=0.5,
    shuffle=True,
    callbacks=[early_stopping]
)

file_label_arrays = file_label_arrays.reshape(nevents,nsamples,1)


print("Data shape: ",data.shape)
print("Labels shape: ",file_label_arrays.shape)
loss = model.evaluate(data, file_label_arrays)
print(f"Test Loss: {loss:.4f}")

plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()



predicted_labels = model.predict(data)


print("Predicted labels shape: ",predicted_labels.shape)

print(predicted_labels[0])

for i in range(len(file_paths)):
    plot_file_2(data[i], predicted_labels[i], file_label_arrays[i])
