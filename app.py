import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt

# Set Matplotlib backend to 'agg' which is suitable for rendering in web apps like Streamlit
plt.switch_backend('agg')


# Function to load and preprocess the data
def load_data():
    # Corrected the filename to match the notebook
    df = pd.read_csv('/content/diabetic_prediabetic_merged.csv')
    df = df[['Age', 'Sex', 'FBS', 'BMI', 'Status']] # Adjusted column names based on the provided dataset.

    df['Age'] = df['Age'].fillna(df['Age'].mean()).astype(int)

    # Fill numerical columns with the mean
    for col in ['FBS', 'BMI']:
      df[col] = df[col].fillna(df[col].mean())

    # Fill categorical columns with the mode
    for col in ['Sex']:
      df[col] = df[col].fillna(df[col].mode()[0])

    return df

# Function to load and preprocess the model
def load_model(X, y):
    # One-hot encode the 'Sex' column
    # Recreate the ColumnTransformer here as it's needed for feature names later
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ['Sex'])], remainder='passthrough')
    X_encoded = ct.fit_transform(X)

    # Convert X_encoded to a float type to ensure compatibility with the model
    X_encoded = X_encoded.astype(float)

    # Encode target variable BEFORE splitting
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42) # Split y_encoded


    # Choose a RandomForestClassifier (well-suited for SHAP)
    shap_model = RandomForestClassifier(random_state=42)

    # Train the model
    shap_model.fit(X_train, y_train) # Train on split y_train

    # Return the trained model, the encoded X, and the test set for SHAP, and X_train, and the label encoder
    # Also return y_test for potential evaluation or plotting later
    return shap_model, X_encoded, X_test, ct, le, X_train, y_test

# Load data
df = load_data()

# Define X and y
X = df[['Age', 'Sex', 'FBS', 'BMI']]
y = df['Status']

# Load and train the model
shap_model, X_encoded, X_test, ct, le, X_train, y_test = load_model(X, y)

st.title("Diabetic Status Prediction")

st.sidebar.header('User Input Features')

# Collect user input features
def user_input_features():
    age = st.sidebar.slider('Age', int(df['Age'].min()), int(df['Age'].max()), int(df['Age'].mean()))
    sex = st.sidebar.selectbox('Sex', df['Sex'].unique())
    fbs = st.sidebar.slider('FBS', float(df['FBS'].min()), float(df['FBS'].max()), float(df['FBS'].mean()))
    bmi = st.sidebar.slider('BMI', float(df['BMI'].min()), float(df['BMI'].max()), float(df['BMI'].mean()))
    data = {'Age': age,
            'Sex': sex,
            'FBS': fbs,
            'BMI': bmi}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Removed display of user input features in the main area
# st.subheader('User Input features')
# st.write(input_df)

# Combine user input features with the original dataset for consistent preprocessing
# This is important to ensure the OneHotEncoder sees all possible categories for 'Sex'
df_for_prediction = pd.concat([input_df, df.drop('Status', axis=1)], axis=0)

# Apply the same ColumnTransformer used for training to the combined data
# Fit on the combined data to ensure all categories are handled
ct_predict = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ['Sex'])], remainder='passthrough')
X_encoded_predict = ct_predict.fit_transform(df_for_prediction)

# The first row of X_encoded_predict is the user input, the rest is the original data
X_user_encoded = X_encoded_predict[0].reshape(1, -1)

# Convert X_user_encoded to float
X_user_encoded = X_user_encoded.astype(float)

# Predict the status
prediction = shap_model.predict(X_user_encoded)
prediction_proba = shap_model.predict_proba(X_user_encoded)

st.subheader('Prediction')
# Decode the prediction
predicted_status = le.inverse_transform(prediction)
st.write(f"The predicted status is: **{predicted_status[0]}**")

st.subheader('Prediction Probability')
# Display prediction probabilities for each class
prediction_proba_df = pd.DataFrame(prediction_proba, columns=le.classes_)
st.write("Probability of each status class:")
st.write(prediction_proba_df)


st.write("---") # Separator

# Removed the SHAP analysis section and force plot


# Optional: Add a section for global explanations like dependence contribution plots (more advanced)
# if isinstance(shap_values, list):
#     st.subheader("SHAP Dependence Contribution Plot")
#     st.write("This plot shows how the interaction between two features affects the prediction.")
#     # Example: Dependence contribution plot for BMI and FBS interaction (if significant)
#     # You need to calculate interaction values first: explainer.shap_interaction_values(X_test)
#     # This is computationally expensive, so only include if necessary and consider sampling X_test
#     # shap.dependence_plot(("BMI", "FBS"), explainer.shap_interaction_values(X_test), X_test_df, feature_names=feature_names_predict, show=False)
#     # st.pyplot(plt.gcf()) # Use plt.gcf() to get the current figure
