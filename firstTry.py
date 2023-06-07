import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Conv1DTranspose
from tensorflow import keras
import pandas as pd
from obspy.io import reftek
from obspy import read, Trace
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
        st = read(file_path, format="REFTEK130",channel_codes=["Z"])
        if not glob.glob(file_path):
            print(f"File not found: {file_path}")
            continue
        z_component = st[0].data
        data.append(z_component)
    return data

def resample_data(data, original_freq, target_freq):
    resampled_data = []
    for sample in data:
        trace = Trace(data=sample, header={"sampling_rate": original_freq})
        resampled_trace = trace.resample(sampling_rate=target_freq)
        resampled_data.append(resampled_trace.data)
    return resampled_data

def apply_highpass_filter(datа, freq, corners):
    filtered_data = []
    for sample in data:
        trace = Trace(data=sample)
        nyquist_freq = 0.5 * freq 
        corner_freq = nyquist_freq / 2  
        trace.filter("highpass", freq=corner_freq,  corners=corners, zerophase=True)
        filtered_data.append(trace.data)
    return filtered_data


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
            label_array = np.zeros(3600*100)
            file_label_arrays.append(label_array)
        else:
            label_array = file_label_arrays[file_paths.index(file_path)]
        event_seconds = int((start_time.minute * 60 + start_time.second)*60)
        label_array[event_seconds] = 1000 if event_type == "led" else 1000

for file_path, label_array in zip(file_paths, file_label_arrays):
    day_number = int(file_path.split("/")[6][-3:])
    day_date = datetime.datetime.strptime(f"2020{day_number:03d}", "%Y%j").date()
    start_time_hh = file_path.split("/")[9].split("_")[0][:2]
    event_count = np.count_nonzero(label_array)
    print(f"File: {file_path}, Day: {day_date}, Start hour: {start_time_hh}h, Events: {event_count}")

data = load_data(file_paths)

#resampled_data = resample_data(data, 100, 60)
filtered_data = apply_highpass_filter(data, freq=1, corners=4)


data = filtered_data

def plot_file(data, label_array):
    fig, ax1 = plt.subplots()
    ax1.plot(data[0])
    ax2 = ax1.twinx()
    ax2.plot(label_array[0],color='red')
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
    ax2.set_ylim([0, 2])
    ax2.legend(loc='upper right')
    plt.show()

#plot_file(data, file_label_arrays)

print("Number of files: ",len(file_paths))
print("Number of label arrays: ",len(file_label_arrays))

data = np.stack(data)
file_label_arrays = np.stack(file_label_arrays)
print("Data initial shape: ",data.shape)
print("Labels initial shape: ",file_label_arrays.shape)
data,file_label_arrays=shuffle(data,file_label_arrays)
nevents,nsamples = data.shape
data = data.reshape(nevents,nsamples,1)
file_label_arrays = file_label_arrays.reshape(nevents,nsamples,1)
print (data.shape)



model = Sequential(
    [
        keras.layers.InputLayer(input_shape=(nsamples,1)),
        keras.layers.Conv1D(filters=32, kernel_size=12, padding="same", strides=1, activation="relu"),
        keras.layers.Conv1D(filters=16, kernel_size=8, padding="same", strides=1, activation="relu"),
        keras.layers.Dropout(rate=0.1),
        keras.layers.Conv1D(filters=8, kernel_size=4, padding="same", strides=1, activation="relu"),
        keras.layers.Conv1DTranspose(filters=8, kernel_size=4, padding="same", strides=1, activation="relu"),
        keras.layers.Dropout(rate=0.1),
        keras.layers.Conv1DTranspose(filters=16, kernel_size=8, padding="same", strides=1, activation="relu"),
        keras.layers.Conv1DTranspose(filters=32, kernel_size=12, padding="same", strides=1, activation="relu"),
        keras.layers.Conv1DTranspose(filters=1, kernel_size=12, padding="same"),
    ]
)

model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.00001))
model.summary()

history = model.fit(
    data,
    file_label_arrays,
    epochs=25,
    batch_size=16,
    validation_split=0.5,
    shuffle=True,
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
