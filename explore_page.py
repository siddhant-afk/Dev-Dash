import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle



def load_model():
    with open("saved_steps.pk1","rb") as file:
        data = pickle.load(file)
    return data



data = load_model()
model = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]


def shorten(categories,cutoff):
    cat_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            cat_map[categories.index[i]] = categories.index[i]
        else:
            cat_map[categories.index[i]] = "Other"

    return cat_map    

def clean_experience(x):
    if x == 'More than 50 years':
        return 50.0
    if x == "Less than 1 year":
        return 0.5
    
    return float(x)

def clean_education(x):
    if "Bachelor’s degree" in x:
        return "Bachelor’s degree"
    if "Master’s degree" in  x:
        return "Master’s degree"
    if "Professional degree" in x or "Other doctoral degree" in x:
        return "Post Grad"
    return "Less than Bachelor's"

def load_data():
    df = pd.read_csv("./Datasets/survey_results_public.csv")
    df = df[["Country", "EdLevel","YearsCode","Employment", "ConvertedCompYearly"]]
    df = df.rename({"ConvertedCompYearly" : "Salary"}, axis=1)
    df = df[df["Salary"].notnull()]
    df.dropna(inplace=True)
    df =df[df["Employment"] == "Employed, full-time"]
    df.drop("Employment",axis=1,inplace=True)

    country_map = shorten(df.Country.value_counts(),400)
    df["Country"] = df["Country"].map(country_map)
    df = df[df["Salary"] <= 300000]
    df = df[df["Salary"] > 10000]
    df = df[df["Country"] != "Other"]

    df["YearsCode"] = df["YearsCode"].apply(clean_experience)
    df["EdLevel"] = df["EdLevel"].apply(clean_education)

    return df

df = load_data()
  
    
def show_explore_page():
    st.title("Explore Software Engineer Salaries")

    st.write(
        """
    ### Stack Overflow Developer Survey 2022
    """
    )

    data = df["Country"].value_counts()

    fig1, ax1 = plt.subplots()
    ax1.pie(data, labels=data.index, autopct="%1.1f%%", shadow=True, startangle=90)
    ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.write("""#### Number of Data from different countries""")

    st.pyplot(fig1)
    
    st.write(
        """
    #### Mean Salary Based On Country
    """
    )

    data = df.groupby(["Country"])["Salary"].mean().sort_values(ascending=True)
    st.bar_chart(data)

    st.write(
        """
    #### Mean Salary Based On Experience
    """
    )

    data = df.groupby(["YearsCode"])["Salary"].mean().sort_values(ascending=True)
    st.line_chart(data)



    st.write("""
    #### Predicted Mean Salary Based on Experience
    """)
    
    df1 = df
    
    df1["EdLevel"] = le_education.fit_transform(df1["EdLevel"])
    df1["Country"] = le_country.fit_transform(df1["Country"])
    

    x = df1.drop("Salary",axis=1)
    y = df1["Salary"]
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)
    
    y_preds = model.predict(x_test)
   

    
    
    new_df = x_test
    new_df["Predicted Salary"] = y_preds
    data1 = new_df.groupby(["YearsCode"])["Predicted Salary"].mean().sort_values(ascending = True )
    st.line_chart(data1)

    st.write("""
    #### Actual Salary VS Predicted Salary
    """)

    fig,ax = plt.subplots()
    ax.plot(df.groupby(["YearsCode"])["Salary"].mean())
    ax.plot(new_df.groupby(["YearsCode"])["Predicted Salary"].mean())

    st.pyplot(fig)



    
    
