import streamlit as st
import pickle
import numpy as np
import sklearn

import warnings
warnings.filterwarnings("ignore")

# Importing the model
model = pickle.load(open("model.pkl", "rb"))

# Importing the dataset/dataframe
data = pickle.load(open("data.pkl", "rb"))

st.title("Laptop Price Predictor")

# User input - Laptop Brand
company = st.selectbox("Brand", data["Company"].unique())

# Use input - Laptop type
type_laptop = st.selectbox("Type", data["TypeName"].unique())

# User input - RAM
RAM = st.selectbox("RAM (in GB)", [2, 4, 6, 8, 12, 16, 24, 32, 64])

# User input - Weight of the laptop
weight = st.number_input("Weight of the laptop")

# User input - Touchscreen
touchscreen = st.selectbox("Touchscreen", ["Yes", "No"])

# User Input - IPS Display
ips = st.selectbox("IPS Display", ["Yes", "No"])

# User Input - Screen Size
screen_size = st.number_input("Screen Size")

# User input - Resolution of the screen
resolution = st.selectbox("Screen Resolution", ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800',
                                                '2880x1800', '2560x1600', '2560x1440', '2304x1440'])

# User input - CPU brand
CPU = st.selectbox("CPU", data["Cpu brand"].unique())

# User input - HDD
HDD = st.selectbox("HDD (in GB)", [0, 128, 256, 512, 1024, 2048])

# User input - SSD
SSD = st.selectbox("SSD (in GB)", [0, 8, 128, 256, 512, 1024])

# User input - GPU brand
GPU = st.selectbox("GPU", data["Gpu brand"].unique())

# User input - Operating Systems
OS = st.selectbox("OS", data["OS"].unique())

# Button for predicting the price of the laptop of the above configuration
if st.button("Predict Price"):

    # Because these cols in data are categorical, taking binary values
    # Thus, converting "Yes"/"No" to 1/0
    touchscreen = 1 if touchscreen == "Yes" else 0
    ips = 1 if ips == "Yes" else 0

    # X- and Y- resolutions are needed to calculate pixels per inch(PPI)
    X_res = int(resolution.split("x")[0])
    Y_res = int(resolution.split("x")[1])
    PPI = np.sqrt(pow(X_res, 2) + pow(Y_res, 2)) / screen_size

    # Input vector, sorta, or query point
    query = np.array([company, type_laptop, RAM, weight, touchscreen, ips, PPI, CPU, HDD, SSD, GPU, OS])
    # Reshaped since it represents a row vector of feature values(12) and thus,
    # an input to the model using which prediction is to be made and back into
    # the original i/p vector
    query = query.reshape(1, 12)

    # Making predictions(Exponential because did log transformation on the target)
    # column. Thus, need to do inverse_transform
    st.title("The predicted price of this configuration is " + str(int(np.exp(model.predict(query))[0])) + " Euros.")
