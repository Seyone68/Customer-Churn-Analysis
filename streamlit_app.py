import shap
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier, Pool
import joblib

import warnings
warnings.filterwarnings("ignore")

# Path of the trained model and data
model_path = 'catboost_model'
data_path = 'Telco-Customer-Churn.csv'

st.set_page_config(page_title='Customer Churn Prediction', page_icon=':bar_chart:', layout='wide')

@st.cache_resource
def load_data():
    data = pd.read_csv(data_path)
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'].fillna(0, inplace=True)
    return data

def load_x_y(file_path):
    data = joblib.load(file_path)
    data.reset_index(drop=True, inplace=True)
    return data

def load_model():
    model = CatBoostClassifier()
    model.load_model(model_path)
    return model

def calculate_shap(model, X_train, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values_cat_train = explainer.shap_values(X_train)
    shap_values_cat_test = explainer.shap_values(X_test)
    return shap_values_cat_train, shap_values_cat_test, explainer

def plot_shap_values(model, shap_values_cat_train, shap_values_cat_test, explainer, X_train, X_test, customer_id):
    # Visualize SHAP values for a specific customer
    customer_index = X_test[X_test['customerID'] == customer_id].index[0]
    fig, ax_2 = plt.subplots(figsize=(6, 6), dpi=200)
    shap.decision_plot(explainer.expected_value, shap_values_cat_test[customer_index], X_test.iloc[customer_index], link='logit', feature_order='hclust', show=False)
    st.pyplot(fig)
    plt.close()

def display_shap_summary(shap_values_cat_train, X_train):
    # Create the plot summarizing the SHAP values
    shap.summary_plot(shap_values_cat_train, X_train, plot_type='bar', plot_size=(12, 12))
    summary_fig, _ = plt.gcf(), plt.gca()
    st.pyplot(summary_fig)
    plt.close()

def display_shap_waterfall_plot(explainer, shap_values, feature_names, max_display=20):
    # Create SHAP waterfall plot
    shap_values_expl = shap.Explanation(values=shap_values, base_values=explainer.expected_value, data=feature_names)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    shap.waterfall_plot(shap_values_expl, max_display=max_display, show=False)
    st.pyplot(fig)
    plt.close()

def summary(model, data, X_train, X_test):
    # Calculate SHAP values
    shap_values_cat_train, shap_values_cat_test, explainer = calculate_shap(model, X_train, X_test)

    # Display SHAP summary plot
    display_shap_summary(shap_values_cat_train, X_train)

def plot_shap(model, X_train, X_test, customer_id):
    # Calculate SHAP values
    shap_values_cat_train, shap_values_cat_test, explainer = calculate_shap(model, X_train, X_test)

    # Display SHAP values for a specific customer
    plot_shap_values(model, shap_values_cat_train, shap_values_cat_test, explainer, X_train, X_test, customer_id)

    # Display SHAP waterfall plot
    customer_index = X_test[X_test['customerID'] == customer_id].index[0]
    display_shap_waterfall_plot(explainer, shap_values_cat_test[customer_index], X_test.columns, max_display=20)

st.title('Telco Customer Churn Prediction')

def main():
    model = load_model()
    data = load_data()

    X_train = load_x_y('X_train.pkl')
    X_test = load_x_y('X_test.pkl')
    y_train = load_x_y('y_train.pkl')
    y_test = load_x_y('y_test.pkl')

    max_tenure = data['tenure'].max()
    max_monthly_charges = data['MonthlyCharges'].max()
    max_total_charges = data['TotalCharges'].max()

    # Radio buttons for selecting the options
    selection = st.radio('Select an option:', ('Feature Importance', 'User-based SHAP', 'Calculate the probability of CHURN'))
    available_customer_ids = X_test['customerID'].tolist()

    if selection == 'User-based SHAP':
        customer_id = st.selectbox('Select a customer ID:', available_customer_ids)
        customer_index = X_test[X_test['customerID'] == customer_id].index[0]
        st.write(f'Customer {customer_id}: Actual value for the Customer Churn: {y_test.iloc[customer_index]}')
        y_pred = model.predict(X_test)
        st.write(f'Customer {customer_id}: Predicted value for the Customer Churn using Catboost Model: {y_pred[customer_index]}')
        plot_shap(model, X_train, X_test, customer_id)

    elif selection == 'Feature Importance':
        summary(model, data, X_train, X_test)

    elif selection == 'Calculate the probability of CHURN':
         # Retrieving data from the user
        customerID = "6464-UIAEA"
        gender = st.selectbox("Gender:", ("Female", "Male"))
        senior_citizen = st.number_input("SeniorCitizen (0: No, 1: Yes)", min_value=0, max_value=1, step=1)
        partner = st.selectbox("Partner:", ("No", "Yes"))
        dependents = st.selectbox("Dependents:", ("No", "Yes"))
        tenure = st.number_input("Tenure:", min_value=0, max_value=max_tenure, step=1)
        phone_service = st.selectbox("PhoneService:", ("No", "Yes"))
        multiple_lines = st.selectbox("MultipleLines:", ("No", "Yes"))
        internet_service = st.selectbox("InternetService:", ("No", "DSL", "Fiber optic"))
        online_security = st.selectbox("OnlineSecurity:", ("No", "Yes"))
        online_backup = st.selectbox("OnlineBackup:", ("No", "Yes"))
        device_protection = st.selectbox("DeviceProtection:", ("No", "Yes"))
        tech_support = st.selectbox("TechSupport:", ("No", "Yes"))
        streaming_tv = st.selectbox("StreamingTV:", ("No", "Yes"))
        streaming_movies = st.selectbox("StreamingMovies:", ("No", "Yes"))
        contract = st.selectbox("Contract:", ("Month-to-month", "One year", "Two year"))
        paperless_billing = st.selectbox("PaperlessBilling", ("No", "Yes"))
        payment_method = st.selectbox("PaymentMethod:", ("Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"))
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=max_monthly_charges, step=0.01)
        total_charges = st.number_input("Total Charges", min_value=0.0, max_value=max_total_charges, step=0.01)
        
        # Confirmation button
        confirmation_button = st.button("Confirm")

        # When the confirmation button is clicked
        if confirmation_button:
            # Create a DataFrame for the input data
            input_data = pd.DataFrame({
                'customerID': [customerID],
                'gender': [gender],
                'SeniorCitizen': [senior_citizen],
                'Partner': [partner],
                'Dependents': [dependents],
                'tenure': [tenure],
                'PhoneService': [phone_service],
                'MultipleLines': [multiple_lines],
                'InternetService': [internet_service],
                'OnlineSecurity': [online_security],
                'OnlineBackup': [online_backup],
                'DeviceProtection': [device_protection],
                'TechSupport': [tech_support],
                'StreamingTV': [streaming_tv],
                'StreamingMovies': [streaming_movies],
                'Contract': [contract],
                'PaperlessBilling': [paperless_billing],
                'PaymentMethod': [payment_method],
                'MonthlyCharges': [monthly_charges],
                'TotalCharges': [total_charges]
            })

            # Predict the probability of churn
            churn_probability = model.predict_proba(input_data)[:, 1]

            # Format the churn probability
            formatted_churn_probability = "{:.2f}".format(churn_probability.item())

            big_text = f"<h1>Churn Probability: {formatted_churn_probability}</h1>"
            st.markdown(big_text, unsafe_allow_html=True)
            st.write(input_data.to_dict())

if __name__ == '__main__':
    main()