import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf



a = tf.keras.models.load_model("finalize_flu_covid_allergy_cold.sav")


def prediction_bean(input_data):
    to_numpy = np.asarray(input_data)
    reshape_data = to_numpy.reshape(1, -1)
    prediction = a.predict(reshape_data)
    final = np.argmax(prediction)
    if final == 0:
        return 'Allergy'
    elif final == 1:
        return 'Cold'
    elif final == 2:
        return 'Covid-19'
    else:
        return 'Flu'

  


def main():
    st.title("Cenos AI")
    inputData = [None] * 20
    #Get input data:
    inputData[0] = st.radio("Cough?", ['Yes', 'No'])
    inputData[1] = st.radio("Muscle Aches?", ['Yes', 'No'])
    inputData[2] = st.radio("Tiredness?", ['Yes', 'No'])
    inputData[3] = st.radio("Sore Throat?", ['Yes', 'No'])
    inputData[4] = st.radio("Runny Nose?", ['Yes', 'No'])
    inputData[5] = st.radio("Stuffy Nose?", ['Yes', 'No'])
    inputData[6] = st.radio("Fever?", ['Yes', 'No'])
    inputData[7] = st.radio("Nausea?", ['Yes', 'No'])
    inputData[8] = st.radio("Vomiting?", ['Yes', 'No'])
    inputData[9] = st.radio("Diarrhea?", ['Yes', 'No'])
    inputData[10] = st.radio("Shortness of Breath?", ['Yes', 'No'])
    inputData[11] = st.radio("Difficulty Breathing?", ['Yes', 'No'])
    inputData[12] = st.radio("Loss of Taste?", ['Yes', 'No'])
    inputData[13] = st.radio("Loss of Smell?", ['Yes', 'No'])
    inputData[14] = st.radio("Itchy Nose?", ['Yes', 'No'])
    inputData[15] = st.radio("Itchy Eyes?", ['Yes', 'No'])
    inputData[16] = st.radio("Itchy Mouth?", ['Yes', 'No'])
    inputData[17] = st.radio("Itchy Inner Ear?", ['Yes', 'No'])
    inputData[18] = st.radio("Sneezing?", ['Yes', 'No'])
    inputData[19] = st.radio("Pink Eye?", ['Yes', 'No'])

    for i in range(20):
        if inputData[i] == 'Yes':
            inputData[i] = 1
        else:
            inputData[i] = 0

    type = 0
    if st.button("Submit"):
        type = prediction_bean(inputData)

    st.success(type)

if __name__ == '__main__':
    main()




