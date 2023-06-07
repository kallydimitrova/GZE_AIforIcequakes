import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Conv1DTranspose
from tensorflow import keras
import pandas as pd
from obspy.io import reftek
from obspy import read
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
import glob

def load_data(file_paths):
    data = []
    for file_path in file_paths:
        st = read(file_path, format="REFTEK130",channel_codes=["Z"])
        if not glob.glob(file_path):
            print(f"File not found: {file_path}")
            continue
        z_component = st[0].data
        data.append(z_component)
    return data

excel_file = pd.read_excel("/home/kalina/ice/Danni_za_Kalina/Events_90-130_Valio.xlsx")

data_dir = "/home/kalina/ice/Danni_za_Kalina/2020/"  
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
            label_array = np.zeros(360000)
            file_label_arrays.append(label_array)
        else:
            label_array = file_label_arrays[file_paths.index(file_path)]
        event_seconds = int((start_time.minute * 60 + start_time.second)*100)
        label_array[event_seconds] = 1 if event_type == "led" else 2

for file_path, label_array in zip(file_paths, file_label_arrays):
    day_number = int(file_path.split("/")[6][-3:])
    day_date = datetime.datetime.strptime(f"2020{day_number:03d}", "%Y%j").date()
    start_time_hh = file_path.split("/")[9].split("_")[0][:2]
    event_count = np.count_nonzero(label_array)
    print(f"File: {file_path}, Day: {day_date}, Start hour: {start_time_hh}h, Events: {event_count}")

data = load_data(file_paths)

def plot_file(data, label_array):
    fig, ax1 = plt.subplots()
    ax1.plot(data[0])
    ax2 = ax1.twinx()
    ax2.plot(label_array[0],color='red')
    ax2.set_ylim([0, 2])
    #plt.ylabel("Amplitude")
    #plt.title("Seismic Data with Event Markers")
    plt.show()

#plot_file(data, file_label_arrays)

print("Number of files: ",len(file_paths))
print("Number of label arrays: ",len(file_label_arrays))

data = np.stack(data)
nevents,nsamples = data.shape
data = data.reshape(nevents,nsamples,1)
print (data.shape)

model = Sequential(
    [
        keras.layers.InputLayer(input_shape=(nsamples,1)),
        keras.layers.Conv1D(filters=16, kernel_size=18, padding="same", strides=1, activation="relu"),
        keras.layers.Conv1D(filters=8, kernel_size=14, padding="same", strides=1, activation="relu"),
        keras.layers.Dropout(rate=0.1),
        keras.layers.Conv1D(filters=4, kernel_size=12, padding="same", strides=1, activation="relu"),
        keras.layers.Conv1DTranspose(filters=4, kernel_size=12, padding="same", strides=1, activation="relu"),
        keras.layers.Dropout(rate=0.1),
        keras.layers.Conv1DTranspose(filters=8, kernel_size=14, padding="same", strides=1, activation="relu"),
        keras.layers.Conv1DTranspose(filters=16, kernel_size=18, padding="same", strides=1, activation="relu"),
        keras.layers.Conv1DTranspose(filters=1, kernel_size=18, padding="same"),
    ]
)

model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(learning_rate=0.00001))
model.summary()

history = model.fit(
    data,
    file_label_arrays,
    epochs=150,
    batch_size=32,
    validation_split=0.5,
    shuffle=True,
)

loss = model.evaluate(data, file_label_arrays)
print(f"Test Loss: {loss:.4f}")

