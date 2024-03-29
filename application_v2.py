import streamlit as st
from src.pipeline.predict_pipeline import CustomData, PredictPipeline


# Page layout
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="📈",
    layout="wide"
)

# User interface
st.header("Student Performance Predictor!")

# User inputs
st.subheader('User Inputs:')
gender = st.selectbox("Gender:", ("male", "female"))
race_ethnicity = st.selectbox("race_ethnicity:", ("group A", "group B", "group C", "group D", "group E"))
parental_level_of_education = st.selectbox(
    "parental_level_of_education:", ("associate's degree", "bachelor's degree", "high school", "master's degree",
                                     "some college", "some high school")
)
lunch = st.selectbox("lunch:", ("free/reduced", "standard"))
test_preparation_course = st.selectbox("test_preparation_course:", ("none", "completed"))
reading_score = st.select_slider("Reading Score:", options=list(range(101)))
writing_score = st.select_slider("Writing Score:", options=list(range(101)))


data = CustomData(
    gender=gender, race_ethnicity=race_ethnicity, parental_level_of_education=parental_level_of_education,lunch=lunch,
    test_preparation_course=test_preparation_course, reading_score=reading_score, writing_score=writing_score
)

pred_df = data.get_data_as_df()
predict_pipeline = PredictPipeline()
results = predict_pipeline.predict(pred_df)

# Prediction
if data and reading_score > 0 and writing_score > 0:
    st.write('The predicted maths score is: ', results[0] if results[0] <= 100 else 100)
