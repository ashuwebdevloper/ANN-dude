import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

st.title("Passenger Survival Prediction")

# -----------------------------
# User Input
# -----------------------------

age = st.slider(
    "Enter the passenger age",
    0,
    100,
    25
)

pclass = st.slider(
    "Enter the passenger class",
    1,
    3,
    1
)

sibsp = st.slider(
    "Enter the number of siblings/spouses aboard",
    0,
    8,
    1
)

parch = st.slider(
    "Enter the number of parents/children aboard",
    0,
    6,
    1
)

fare = st.slider(
    "Enter the fare paid by the passenger",
    0.0,
    500.0,
    7.25
)

sex = st.selectbox(
    "Enter passenger sex",
    ["male", "female"]
)

embarked = st.selectbox(
    "Enter the port of embarkation",
    ["C", "Q", "S"]
)


# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict"):

    # Create input DataFrame
    data = pd.DataFrame([
        {
            'Pclass': pclass,
            'Age': age,
            'SibSp': sibsp,
            'Parch': parch,
            'Fare': fare,
            'Sex': sex,
            'Embarked': embarked
        }
    ])

    st.write("Input Data:")
    st.dataframe(data)

    # -----------------------------
    # Load model
    # -----------------------------

    model = load_model("model.h5")

    # Load scaler
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Load one-hot encoder
    with open("onehot_encoder.pkl", "rb") as f:
        onehot_encoder = pickle.load(f)

    # Load label encoder
    with open("lable_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)


    # -----------------------------
    # Encode Sex
    # -----------------------------

    data["Sex"] = label_encoder.transform(data["Sex"])


    # -----------------------------
    # Encode Embarked
    # -----------------------------

    embarked_encoded = onehot_encoder.transform(
        data[["Embarked"]]
    )

    embarked_df = pd.DataFrame(
        embarked_encoded,
        columns=onehot_encoder.get_feature_names_out(["Embarked"]),
        index=data.index
    )

    data = pd.concat(
        [
            data.drop("Embarked", axis=1),
            embarked_df
        ],
        axis=1
    )


    # -----------------------------
    # Scale numerical columns
    # -----------------------------

    data[
        ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    ] = scaler.transform(
        data[
            ["Pclass", "Age", "SibSp", "Parch", "Fare"]
        ]
    )


    # -----------------------------
    # Prediction
    # -----------------------------

    prediction = model.predict(data)

    probability = prediction[0][0]


    # -----------------------------
    # Result
    # -----------------------------

    if probability >= 0.5:
        st.success(
            "The passenger is predicted to survive."
        )
    else:
        st.error(
            "The passenger is predicted not to survive."
        )

    st.write(
        f"Survival probability: {probability:.2%}"
    )