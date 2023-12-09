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
import pickle
import streamlit as st

# Set default template for Plotly
pio.templates.default = "plotly_white"

# Load the dataset
data = pd.read_csv("ad_10000records.csv")

# Visualizations Section

# Heatmap
# data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})
# heatmap = sns.heatmap(data.corr(), cmap="coolwarm", annot=True)
    
# Map binary target variable to Yes/No for better visualization
data["Clicked on Ad"] = data["Clicked on Ad"].map({0: "No", 1: "Yes"})
# data["Gender"] = data["Gender"].map({1: "Male", 0: "Female"})

## Clickthrough rate based on Gender
fig_gender = plt.figure()
sns.countplot(data=data, x='Clicked on Ad', hue='Gender')

# Model Building Section
## Convert categorical variables to numeric
data["Clicked on Ad"] = data["Clicked on Ad"].map({"No": 0, "Yes": 1})
data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})

## Split data into features and target variable
x , y = data.drop(["Ad Topic Line", "City", "Country", "Timestamp", "Clicked on Ad"],axis=1), data['Clicked on Ad']

## Split data into training and testing sets
X_train , X_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=0)

## Build a RandomForestClassifier model
model = pickle.load(open("rf_model.pkl", 'rb'))
model.fit(X_train, y_train)

## Predictions on the test set
y_pred = model.predict(X_test)

## Model evaluation - Accuracy
accuracy = accuracy_score(y_test, y_pred)

## Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Sidebar
st.sidebar.write("Ads Click Through Rate(CTR) helps CMO's and advertisers understand how well their ads are performing and how well users are engaging with the content the brand is pushing.")
st.sidebar.write("Ads Click Through Rate is the ratio of how many users clicked on your ad to how many users viewed your ad. For example, 5 out of 100 users click on the ad while watching a youtube video. So, in this case, the CTR of the youtube ad will be 5%. Analyzing the click-through rate help companies in finding the best ad for their target audience.")
st.sidebar.write("Ads Click-through rate prediction means predicting whether the user will click on the ad.")
st.sidebar.write("Let's dive in, analyse our ads data and make a predictor model!")
st.markdown("# Ads Click Through Rate Prediction")
# Create three tabs with different names
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Introduction", "Visualization", "Model Building", "Predictor", "About Me"])

# Add elements to each tab using with notation
with tab1:
    st.image("https://s3.amazonaws.com/newblog.psd2html.com/wp-content/uploads/2021/01/12115609/what-is-banner-ctr.png")

    st.markdown('---')
    # Dataset display
    # st.title("Introduction")
    st.markdown("We have this data sourced from [statso.io](https://docs.streamlit.io/library/api-reference/widgets/st.link_button). We will be build our prediction model using this database.")

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
        - With minimum being \$13,996 and maximum being \$79,332
    - Daily Internet Usage:
        - On average, each user spends about 177 mins browsing the internet
        - The least time being 105 mins and highest being 270 mins
    """)

with tab2:
    st.image("https://www.grazitti.com/assets/2017/01/main.png")
    st.markdown("---")
    
    ## Visualization Section
    st.title("Click Through Rate Analysis")
    st.markdown("---")


    st.subheader("Clickthrough Rate Based on Gender")
    st.pyplot(fig_gender)
    st.write("From the above graph, we can see that Females click more on ads than Males")
    st.markdown("---")

    st.write("Select a variable for visualisation")
    cols = ['Daily Time Spent on Site', 'Daily Internet Usage', 'Age', 'Area Income']
    var = st.selectbox("Please pick one variable", cols)
    fig_var = px.box(data, x=var, color="Clicked on Ad", 
                 color_discrete_map={'Yes':'blue', 'No':'red'})
    fig_var.update_traces(quartilemethod="exclusive")
    st.plotly_chart(fig_var)
    if var == "Daily Time Spent on Site":
        st.write("From the above graph, we can see that the users who spend more time on the website click more on ads")
        st.markdown('---')
    if var == "Daily Internet Usage":
        st.write("We can see that the users with high internet usage click less on ads compared to the users with low internet usage")
        st.markdown('---')
    if var == "Age":
        st.write("We can see that users around 40 years click more on ads compared to users around 27-36 years old")
        st.markdown('---')
    if var == "Area Income":
        st.write("There’s not much difference, but people from high-income areas click less on ads.")
        st.markdown("---")
    
    # # Heatmap
    # st.pyplot(heatmap.figure)
    # st.write("This heatmap provides a visual represtation of the correlations between the features")
    # st.markdown("---")

with tab3:
    st.image("ml.jpg")
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
    
    st.subheader("Model Evaluation")
    st.write(f"Accuracy: {accuracy:.2%}")
    st.write("Model evaluation is the process of measuring how well a machine learning model performs on a given task. One of the common metrics for model evaluation is accuracy, which is the ratio of correct predictions to total predictions. Accuracy can range from 0% to 100%, where 100% means perfect prediction and 0% means no prediction.")
    st.markdown('---')

    # Confusion matrix
    st.subheader("Confusion Matrix")

    # Create a figure object
    fig = plt.figure()

    # Plot the heatmap on the figure
    sns.heatmap(cm, fmt='g', cbar=False, annot=True)

    # Set the labels and title on the figure
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")

    # Display the figure using st.pyplot
    st.pyplot(fig)
    st.write("A confusion matrix is a table that shows how well a machine learning model performs on a set of test data with known labels. It compares the actual labels with the predicted labels and counts the number of correct and incorrect predictions for each class. A confusion matrix can help evaluate the accuracy, precision, recall, and other metrics of a classification model.")
    st.write("""
        - *True Positive (TP)*:  This is the number of instances that are actually clicked and predicted as clicked by the model. For example, if the task is to predict whether a user will click on an ad, then TP is the number of users who clicked on the ad and were correctly predicted as clickers by the model. The TP value here is 823.
        - *False Positive (FP)*: This is the number of instances that are actually no clicked but predicted as clicked by the model. This is also known as a Type I error. For example, if the task is to predict whether a user will click on an ad, then FP is the number of users who did not click on the ad but were wrongly predicted as clickers by the model. The FP value here is 166.
        - *False Negative (FN)*: This is the number of instances that are actually clicked but predicted as no clicked by the model. This is also known as a Type II error. For example, if the task is to predict whether a user will click on an ad, then FN is the number of users who clicked on the ad but were wrongly predicted as non-clickers by the model. The TN value here is 227
        - *True Negative (TN)*: This is the number of instances that are actually no clicked and predicted as no clicked by the model. For example, if the task is to predict whether a user will click on an ad, then TN is the number of users who did not click on the ad and were correctly predicted as non-clickers by the model. The TN value here is 784.
        
        The confusion matrix can help us understand the strengths and weaknesses of a classification model, as well as the trade-offs between different performance measures. For instance, a model that has a high precision may have a low recall, or vice versa. A model that has a high accuracy may not be very sensitive or specific, or vice versa. A good model should balance these metrics according to the needs and goals of the problem.
    """)

with tab4:
    st.image("https://cdn.dribbble.com/users/579758/screenshots/5546963/18-11-25-s.jpg")

    ## User Input Section
    st.title("Ads Click Through Rate Prediction")

    # Get user inputs
    daily_time_spent = st.number_input("Daily Time Spent on Site", min_value=0, format="%d", step=10)
    age = st.number_input("Age", min_value=0, format="%d", step=1)
    area_income = st.number_input("Area Income", min_value=0, format="%d", step=1000)
    daily_internet_usage = st.number_input("Daily Internet Usage", min_value=0, format="%d", step=10)
    gender = st.radio("Gender (Male = 1, Female = 0)", options=[1, 0])

    # Make predictions
    user_features = np.array([[daily_time_spent, age, area_income, daily_internet_usage, gender]])
    prediction = model.predict(user_features)

    st.markdown("<style>div.stButton > button:first-child {background-color: rgb(255, 255, 255); color: rgb(0, 0, 0);}</style>", unsafe_allow_html=True)

    # Create a button with the prediction result and a custom label
    st.write("Will the user click on ad?")
    st.button(f"{'👍 Yes' if prediction[0] == 1 else '👎 No'}")

with tab5:
    st.image("myself.jpg")
    st.markdown("---")
    st.write("""
        Hello everyone! I'm Vishal, a data science enthusiast who loves to play with data and uncover hidden patterns and insights. I enjoy using programming languages such as Python and R to analyze data and create interactive dashboards and web apps. I am always curious to learn new skills and techniques to keep up with the fast-paced and dynamic field of data science.

        Apart from data analysis, I have a passion for sports and gaming. One of my favorite hobbies is playing football with my friends. I love the thrill and excitement of the game, as well as the teamwork and camaraderie that it fosters. Playing football also helps me stay fit and healthy, which is important for me. Another way I like to keep myself fit is by going to the gym regularly. I find working out to be a great way to relieve stress and improve my mood. It also challenges me to push myself beyond my limits and achieve my fitness goals.

        When I'm not playing football or working out, I like to indulge in some gaming. Gaming is not just a fun activity for me; it's also a way to stimulate my mind and imagination. I enjoy playing different genres of games, from action-adventure to strategy to simulation. Gaming allows me to explore different worlds and scenarios, and also to interact with other players and make new friends.

        I invite you to check out my web app and join me on this amazing journey of data discovery and analytics.
    """)
