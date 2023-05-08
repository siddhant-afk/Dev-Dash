import streamlit as st
import pickle
import numpy as np


def load_model():
    with open("saved_steps.pk1","rb") as file:
        data = pickle.load(file)
    return data



data = load_model()
model = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]


def show_predict_page():
    st.title("Software Developer Salary Prediction")

    st.write("""#### We need some information to predict the salary""")

    countries =  ("United States of America",
                  "Germany",
                  "United Kingdom of Great Britain and Northern Ireland",
                  "India",
                  "Canada",
                  "France",
                  "Brazil",
                  "Spain",
                  "Netherlands",
                  "Australia",
                  "Italy",
                  "Poland",
                  "Sweden",
                  "Russian Federation",
                  "Switzerland")
    
    education = ("Master’s degree",
                 "Bachelor’s degree",
                 "Post Grad",
                 "Less than Bachelor's")
    
    country = st.selectbox("Country",countries)
    edlevel = st.selectbox("Education",education)
    experience = st.slider("Years of Experince",0,50,3)
    ok = st.button("Calculate Salary")

    if ok:
        x = np.array([[country,edlevel,experience]])
        x[:,0] = le_country.transform(x[:,0])
        x[:,1] = le_education.transform(x[:,1])
        x = x.astype(float)

        salary = model.predict(x)
        st.subheader(f"The estimated salary is : ${salary[0]:.2f}")
