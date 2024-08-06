
# Drug Effectiveness Predictor

This project is a Streamlit application designed to predict the effectiveness of a drug based on its name and the condition it treats. The application uses an XGBoost model to make these predictions. 

*Note:* The predictions are based on a simplified model and should be taken with caution. The model uses limited information and random values for many features, which may affect the accuracy of the predictions.

## Features

- Predicts drug effectiveness based on the drug name and the condition.
- Provides a visual representation of the predicted effectiveness.
- Includes a sidebar with information about the limitations of the model.

## Installation

### Prerequisites

- Python 3.6 or higher
- Streamlit
- NumPy
- pandas
- joblib
- xgboost
- matplotlib

### Steps

1. Clone the repository:
   bash
    git clone <[repository-url](https://github.com/francesseyram/Intro-To-AI-Final-Project.git)>
    cd <[[repository-folder](https://github.com/francesseyram/Intro-To-AI-Final-Project/tree/main)]>

2. Install the required packages:
   bash
   pip install -r requirements.txt
   

3. Ensure you have the following files in your project directory:
   - xgboost_model.joblib: The pre-trained XGBoost model.
   - Drug.csv: The dataset containing drug names and conditions.

4. Run the Streamlit application:
   bash
   streamlit run app.py
   

## Usage

1. Open the application in your web browser. The default URL is http://localhost:8501.
2. Enter the drug name and the condition in the input fields.
3. Click the "Predict" button to see the predicted effectiveness.

## Hosting

### Local Server

To host the application on a local server, follow the installation steps above and run the application using the streamlit run app.py command.

### Cloud Deployment

You can deploy the application on various cloud platforms like Heroku, AWS, GCP, or Azure. Here is a general guide using Heroku:

1. Install the Heroku CLI:
   bash
   curl https://cli-assets.heroku.com/install.sh | sh
   

2. Log in to your Heroku account:
   bash
   heroku login
   

3. Create a new Heroku app:
   bash
   heroku create your-app-name
   

4. Add a Procfile to your project directory with the following content:
   
   web: streamlit run app.py
   

5. Commit your changes:
   bash
   git add .
   git commit -m "Initial commit"
   

6. Deploy the app to Heroku:
   bash
   git push heroku main
   

7. Open your app in the browser:
   bash
   heroku open
   

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- This application uses the XGBoost library for machine learning.
- Streamlit is used for creating the web interface.
- Thanks to the contributors of the open-source libraries used in this project.
