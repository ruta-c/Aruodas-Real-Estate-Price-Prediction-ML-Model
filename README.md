# Aruodas-Real-Estate-Price-Prediction-ML-Model
This project entails the extraction of data from real estate advertisements on the web page aruodas.lt, followed by data preparation and the development of a machine learning model for price prediction. The primary objective of this endeavor is exclusively educational, aimed at fostering a deeper understanding of data extraction, preparation, and predictive modeling within the context of real estate pricing.

The model performance and practical implementation can be found here: https://price-predictor-hdcz.onrender.com

Data analysis: https://app.powerbi.com/groups/me/reports/3363f855-0049-4131-9f4a-b331082b2714?ctid=91d0140d-3d7e-4fc3-9489-69cf0d469a3f&pbi_source=linkShare

![Aruodas (1)](https://github.com/ruta-c/Aruodas-Real-Estate-Price-Prediction-ML-Model/assets/130843221/4f558225-ee3c-4bb9-ba9b-8b09c4dc3815)

# Project Structure
* Aruodas Scraper
  * The scraping module focused on extracting data from Aruodas.lt using Selenium. Selenium was chosen for its compatibility with the webpage structure. This phase involves the insertion of new advertisement data into an AWS RDS (Relational Database Service).
* Data Cleaning
  * Responsible for data preparation tasks, including cleaning and standardizing data types. This module ensures the data is in a suitable format for machine learning modeling.
* Model Setup
  * Configuration of the machine learning model. In this section, the RandomForestRegressor was chosen as the preferred model based on its superior performance compared to other models such as GradientBoostingRegressor and LinearRegression.
* Training
  * Conducting the training process for the machine learning model. This involves feeding the prepared data into the model. The trained model is then serialized and stored using the joblib library.
* Layout
  * Preparation of the layout for the Dash web application. This module involves defining the structure and appearance of the user interface, ensuring a user-friendly and intuitive experience.
* Dash App
  * The main file for the Dash web application. This includes the app function and callback functions, which handle the logic of the application. The Dash app serves as the user interface for interacting with the machine learning model and visualizing the results.
  
