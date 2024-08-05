# Intro-To-AI-Final-Project
# Drug Effectiveness and Ease of Use Prediction

This Streamlit app predicts drug effectiveness and ease of use based on user input for drug name and condition. It uses machine learning models to provide predictions and visualizations.

## Features

- Predict drug effectiveness and ease of use.
- Visualize predictions with bar charts.
- Handle user inputs for drug name and condition.

## Prerequisites

Before running the app, ensure you have the following installed:

- Python 3.x
- Pip (Python package installer)

## Installation

1. *Clone the Repository:*

    bash
    git clone <[repository-url](https://github.com/francesseyram/Intro-To-AI-Final-Project.git)>
    cd <[[repository-folder](https://github.com/francesseyram/Intro-To-AI-Final-Project/tree/main)]>
    

2. *Create a Virtual Environment (optional but recommended):*

    bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    

3. *Install Dependencies:*

    Ensure you have a requirements.txt file (included below) that lists all required Python packages. Install them using:

    bash
    pip install -r requirements.txt
    

4. *Download or Create Models and Preprocessors:*

    Make sure you have the trained models and preprocessors saved as .joblib files in the root directory:
    - rf_model_effective.joblib
    - label_encoders.joblib
    - scaler.joblib

5. *Prepare the Data:*

    Ensure the Drug.csv file is available in the root directory. This file should contain columns Drug, Condition, and Reviews.

6. *Run the App:*

    Start the Streamlit app using the following command:

    bash
    streamlit run app.py
    

    The app will be available at http://localhost:8501 by default.

## Cloud Hosting

To host the app on the cloud, you can use platforms like [Streamlit Sharing](https://streamlit.io/sharing), [Heroku](https://www.heroku.com/), or [Google Cloud Platform](https://cloud.google.com/). For Streamlit Sharing, follow the instructions on their website to deploy your app directly from a GitHub repository.

## Usage

1. Open the app in your web browser.
2. Enter the drug name and condition in the provided text inputs.
3. Click the "Predict" button to get the results.
4. View the predictions and the visualization of model predictions.

## Troubleshooting

- If you encounter issues with missing files or models, ensure all required .joblib files are present and correctly named.
- If the app does not run, check for errors in the terminal and ensure all dependencies are installed.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
