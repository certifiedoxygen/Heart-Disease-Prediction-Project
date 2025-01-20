import streamlit as st
import joblib
import numpy as np
import pandas as pd
from PIL import Image

icon = Image.open("heart.png") 
st.set_page_config(
    page_title="Heart Disease Prediction", 
    page_icon=icon 
)


age_groups = {
    1: "18-24 years old",
    2: "25-29 years old",
    3: "30-34 years old",
    4: "35-39 years old",
    5: "40-44 years old",
    6: "45-49 years old",
    7: "50-54 years old",
    8: "55-59 years old",
    9: "60-64 years old",
    10: "65-69 years old",
    11: "70-74 years old",
    12: "75-79 years old",
    13: "80 years and older"
}

quality_ratings = {
    1: "Excellent",
    2: "Very Good",
    3: "Good",
    4: "Fair",
    5: "Poor",
}

gender = {
    0: 'Female',
    1: 'Male'
}

education_levels = {
    1: "Never attended school or only kindergarten",
    2: "Grades 1 through 8 (Elementary)",
    3: "Grades 9 through 11 (Some high school)",
    4: "Grade 12 or GED (High school graduate)",
    5: "College 1 year to 3 years (Some college or technical school)",
    6: "College 4 years or more (College graduate)"
}

yesno = {
    0: "No",
    1: "Yes"
}

income_ranges = {
    1: "<10,000",
    2: "10,000–15,000",
    3: "15,000–20,000",
    4: "20,000–25,000",
    5: "25,000–35,000",
    6: "35,000–50,000",
    7: "50,001–75,000",
    8: ">75,000"
}

log_tuned = joblib.load('LogisticRegressionModel')
scaler = joblib.load('Scaler_')

columns =   ['PhysHlth',
 'MentHlth',
 'Age',
 'Diabetes',
 'Stroke',
 'DiffWalk',
 'GenHlth',
 'HighBP',
 'HighChol',
 'Income',
 'Smoker',
 'BMI',
 'Sex',
 'Education',
 'PhysActivity']


def main():

    st.markdown("## Heart Disease Prediction 🫀")

    PhysHlth = st.slider("For how many days during the past 30 days was your Physical Health not good?", min_value = 0, max_value = 30, step = 1)
    MentHlth = st.slider("For how many days during the past 30 days was your Mental Health not good?", min_value = 0, max_value = 30, step = 1)
    Education = st.selectbox("What is the highest grade or year of school you completed?", options = list(education_levels.keys()), format_func = lambda x: education_levels[x])



    col1, col2= st.columns(2)
    
    Age = col1.selectbox("What is your Age?", options = list(age_groups.keys()), format_func = lambda x: age_groups[x])
    Sex = col1.selectbox("What is your Gender", options = list(gender.keys()), format_func = lambda x: gender[x])
    Income = col1.selectbox("What is your annual household income?", options = list(income_ranges.keys()), format_func = lambda x: income_ranges[x])

    Diabetes = col1.radio("Do you have Diabetes?", options = list(yesno.keys()), format_func = lambda x: yesno[x])
    Stroke = col1.radio("Have you ever had a Stroke?", options = list(yesno.keys()), format_func = lambda x: yesno[x])
    GenHlth = col2.selectbox("How would you rate your General Health?", options = list(quality_ratings.keys()), format_func = lambda x: quality_ratings[x])
    BMI = col2.number_input("What is your BMI?", min_value = 5, max_value = 50, value = 25, step = 1)

    DiffWalk = col2.radio("Do you have serious difficulty walking or climbing stairs?", options = list(yesno.keys()), format_func = lambda x: yesno[x])
    HighBP = col2.radio("Do you have high blood pressure?", options = list(yesno.keys()), format_func = lambda x: yesno[x])
    HighCol = col2.radio("Do you have high Cholestrol", options = list(yesno.keys()), format_func = lambda x: yesno[x])
    Smoker = col1.radio("Have you had atleast 100 cigarettes in your entire life?", options = list(yesno.keys()), format_func = lambda x: yesno[x])
    PhysActivity = col2.radio("Have you engaged in Physical Activity in the past 30 days?", options = list(yesno.keys()), format_func = lambda x: yesno[x])

    def predict():
        row = np.array([PhysHlth, MentHlth, Age, Diabetes, Stroke, DiffWalk, GenHlth, HighBP, HighCol, Income, Smoker, BMI, Sex, Education, PhysActivity])
        X = pd.DataFrame([row], columns = columns)
        print("Input DataFrame:\n", X)
        print("Shape of Input DataFrame:", X.shape)
        X_scaled = scaler.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns = columns)
        print("Scaled Data:\n", X_scaled_df)
        print("Shape of Scaled Data:", X_scaled.shape)
        prediction  = log_tuned.predict_proba(X_scaled)[:, 1][0]

        with result_container:
            if prediction < 0.5:
                risk_level = "Low Risk of Heart Disease"
                risk_color = "#40916c"
            elif 0.5 <= prediction < 0.8:
                risk_level = "Moderate Risk of Heart Disease"
                risk_color = "#e26d5c"
            else:
                risk_level = "High Risk of Heart Disease"
                risk_color = "#9e2a2b"


            st.markdown(f"""
            <div style="
                background-color: {risk_color};
                padding: 8px;
                border-radius: 8px;
                text-align: left;
                font-family: Arial, sans-serif;
                box-shadow: 1px 1px 5px rgba(0, 0, 0, 0.1);
                line-height: 1.2;
            ">
                <p style="color: #fff; margin: 0; font-size: 16px; font-weight: 600;">{risk_level}</p>
                <p style="color: #fff; margin: 2px 0 0; font-size: 14px; font-weight: 400;">Probability: {prediction:.3f}</p>
            </div>
            """, unsafe_allow_html=True)




    calc = st.button("Check Result", on_click = predict)
    result_container = st.container()




if __name__ == "__main__":
    main()