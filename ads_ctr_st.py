# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import streamlit as st

# Set default template for Plotly
pio.templates.default = "plotly_white"

# Load the dataset
data = pd.read_csv("ad_10000records.csv")

# Map binary target variable to Yes/No for better visualization
data["Clicked on Ad"] = data["Clicked on Ad"].map({0: "No", 1: "Yes"})

# Visualizations Section

## Clickthrough rate based on Gender
fig_gender = plt.figure()
sns.countplot(data=data, x='Clicked on Ad', hue='Gender')

## Time Spent on Site vs. Click Through Rate
fig_time_spent = px.box(data, x="Daily Time Spent on Site", color="Clicked on Ad",
                        color_discrete_map={'Yes':'blue', 'No':'red'})
fig_time_spent.update_traces(quartilemethod="exclusive")

## Daily Internet Usage vs. Click Through Rate
fig_internet_usage = px.box(data, x="Daily Internet Usage", color="Clicked on Ad", 
                            color_discrete_map={'Yes':'blue', 'No':'red'})
fig_internet_usage.update_traces(quartilemethod="exclusive")

## Age vs. Click Through Rate
fig_age = px.box(data, x="Age", color="Clicked on Ad", 
                 color_discrete_map={'Yes':'blue', 'No':'red'})
fig_age.update_traces(quartilemethod="exclusive")

## Area Income vs. Click Through Rate
fig_income = px.box(data, x="Area Income", color="Clicked on Ad", 
                    color_discrete_map={'Yes':'blue', 'No':'red'})
fig_income.update_traces(quartilemethod="exclusive")

# Model Building Section
## Convert categorical variable 'Gender' to numeric
data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})

## Split data into features and target variable
x = data.iloc[:, 0:7].drop(['Ad Topic Line', 'City'], axis=1)
y = data.iloc[:, 9]

## Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

## Build a RandomForestClassifier model
model = RandomForestClassifier()
model.fit(x_train, y_train)

## Predictions on the test set
y_pred = model.predict(x_test)

## Model evaluation - Accuracy
accuracy = accuracy_score(y_test, y_pred)

## Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Create three tabs with different names
tab1, tab2, tab3, tab4 = st.tabs(["Intro", "Visualization", "Owl", "Input"])

# Add elements to each tab using with notation
with tab1:
    st.header("Introduction")
    st.image("https://static.streamlit.io/examples/cat.jpg", width=200)

    st.markdown('---')
    # Dataset display
    # st.title("Introduction")
    st.markdown("We have this data sourced from [statso.io](https://docs.streamlit.io/library/api-reference/widgets/st.link_button). We will be build our prediction model using this database.")

    ## Convert categorical variable 'Gender' to numeric
    data["Gender"] = data["Gender"].map({1:"Male" , 0:"Female"})
    st.dataframe(data)
    st.write("The shape of our dataset is 10,000 x 10")
    st.write("Which means we have 10,000 rows with 10 column attributes")

    st.markdown('---')
    st.write("Here is a brief statistical description of our data")
    st.dataframe(data.describe().T, use_container_width=True)
    st.write("From the above table we can observe that:")
    st.markdown("""
    - Daily time spent on site:
        - On an average users spend about 61 mins on the website browsing
        - With minimum being 32 mins and max being 91 mins
    - Age:
        - The mean age of users is 35 years
        - With the youngest being 19 and eldest being 60
    - Area Income:
        - The mean income per area is $53,840
        - With minimum being $13,996 and maximum being $79,332
    - Daily Internet Usage:
        - On average, each user spends about 177 mins browsing the internet
        - The least time being 105 mins and highest being 270 mins
    """)

with tab2:
    st.header("Now let's analyze some important features")
    st.image("https://static.streamlit.io/examples/dog.jpg", width=200)
    st.markdown("---")
    
    ## Visualization Section
    st.title("Click Through Rate Analysis")
    st.markdown("---")


    st.subheader("Clickthrough Rate Based on Gender")
    st.pyplot(fig_gender)
    st.write("From the above graph, we can see that Females click more on ads than Males")
    st.markdown("---")
 
    st.subheader("Time Spent on Site vs. Click Through Rate")
    st.plotly_chart(fig_time_spent)
    st.write("From the above graph, we can see that the users who spend more time on the website click more on ads")
    st.markdown('---')

    st.subheader("Daily Internet Usage vs. Click Through Rate")
    st.plotly_chart(fig_internet_usage)
    st.write("We can see that the users with high internet usage click less on ads compared to the users with low internet usage")
    st.markdown('---')

    st.subheader("Age vs. Click Through Rate")
    st.plotly_chart(fig_age)
    st.write("We can see that users around 40 years click more on ads compared to users around 27-36 years old")
    st.markdown('---')

    st.subheader("Area Income vs. Click Through Rate")
    st.plotly_chart(fig_income)
    st.write("There’s not much difference, but people from high-income areas click less on ads.")

with tab3:
    st.header("Model Building")
    st.image("https://static.streamlit.io/examples/owl.jpg", width=200)
    st.markdown("---")
    
    ## Model Building Section
    st.title("Click Through Rate Prediction Model")
    st.markdown("---")
    st.subheader("Random Forest Classifier")
    st.image("https://www.researchgate.net/publication/334365773/figure/download/fig3/AS:779093964513281@1562761884487/Ensemble-classifier-random-forest-classifier-general-concept-6.png")
    st.write("""
        A random forest classifier is a machine learning algorithm that can be used to solve both classification and regression problems. It works by creating a group of decision trees, each of which is trained on a random subset of the data and features. A decision tree is a simple way of making predictions based on a series of questions or rules. For example, if you want to predict whether a person will buy a product or not, you can ask questions like “How old are they?”, “What is their income?”, “How often do they shop online?”, etc. Each question will split the data into two or more branches, until you reach a final answer or prediction. In a similar manner, we will build a classifier to predict whether the user will click on the ad or not.

        A random forest classifier combines the predictions of multiple decision trees to produce a more accurate and robust result. It does this by taking the majority vote or the average of the predictions from all the trees. This way, it can reduce the errors and biases that might occur in a single decision tree. A random forest classifier can also handle missing values, outliers, and imbalanced data. It can also rank the importance of the features that affect the prediction.
    """)
    st.markdown("---")

    # st.subheader("Model Evaluation")
    # st.write(f"Accuracy: {accuracy:.2%}")
    # st.markdown('---')

    # st.subheader("Confusion Matrix")
  
    # # Create a figure object
    # fig = plt.figure()

    # # Plot the heatmap on the figure
    # sns.heatmap(cm, fmt='g', cbar=False, annot=True)

    # # Set the labels and title on the figure
    # plt.ylabel("Actual")
    # plt.xlabel("Predicted")
    # plt.title("Confusion Matrix")

    # # Display the figure using st.pyplot
    # st.pyplot(fig)

with tab4:
    st.header("User Input")

    ## User Input Section
    st.title("Ads Click Through Rate Prediction")

    # Get user inputs
    daily_time_spent = st.number_input("Daily Time Spent on Site", min_value=0.0, format="%f")
    age = st.number_input("Age", min_value=0)
    area_income = st.number_input("Area Income", min_value=0.0, format="%f")
    daily_internet_usage = st.number_input("Daily Internet Usage", min_value=0.0, format="%f")
    gender = st.radio("Gender (Male = 1, Female = 0)", options=[1, 0])

    # Make predictions
    user_features = np.array([[daily_time_spent, age, area_income, daily_internet_usage, gender]])
    prediction = model.predict(user_features)

    # st.write(f"Will the user click on ad? {'Yes' if prediction[0] == 1 else 'No'}")
    st.markdown("<style>div.stButton > button:first-child {background-color: rgb(255, 255, 255); color: rgb(0, 0, 0);}</style>", unsafe_allow_html=True)

    # Create a button with the prediction result and a custom label
    st.write("Will the user click on ad?")
    st.button(f"{'👍 Yes' if prediction[0] == 1 else '👎 No'}")

