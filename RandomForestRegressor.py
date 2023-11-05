from sqlalchemy import create_engine
import dash
from dash import dcc, html, Output, Input
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load your real estate dataset here and preprocess if necessary
engine = create_engine('postgresql://postgres:odievai1@localhost:5432/aruodas')
sql_query = 'SELECT * FROM cleaned_flats'
df = pd.read_sql_query(sql_query, engine)

# Define features and target variable
features = ['area', 'rooms', 'floor', 'floors',
       'type_Blokinis', 'type_Kita', 'type_Medinis',
       'type_Monolitinis', 'type_Mūrinis', 'mounting_Dalinė apdaila',
       'mounting_Kita', 'mounting_Neįrengtas', 'mounting_Įrengtas',
       'energy_class_A', 'energy_class_A+', 'energy_class_A++',
       'energy_class_B', 'energy_class_Lower than B', 'aeroterminis',
       'dujinis', 'elektra', 'geoterminis', 'kietu kuru', 'sildymas_kita',
       'saulės energija', 'skystu kuru', 'centrinis_sildymas', 'balkonas',
       'drabužinė', 'none', 'palepe', 'pirtis', 'rūsys', 'sandėliukas',
       'terasa', 'vieta_automobiliui', 'kameros', 'kodine_spyna', 'sargas',
       'sarvuotos_durys', 'signalizacija', 'atskiras įėjimas', 'aukcionas',
       'aukštos lubos', 'butas palėpėje', 'butas per kelis aukštus',
       'buto dalis', 'internetas', 'kabelinė televizija',
       'nauja elektros instaliacija', 'nauja kanalizacija',
       'tualetas ir vonia atskirai', 'uždaras kiemas',
       'virtuvė sujungta su kambariu', 'building_age', 'building_age_reno',
       'distance_to_center']

# Split the dataset into features and target variable
X = df[features]
y = df['price_sqm']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the RandomForestRegressor model
model = RandomForestRegressor(min_samples_split=2, n_estimators=739, random_state=42)
model.fit(X_train, y_train)

# Predict prices for the test set
y_pred = model.predict(X_test)

# Create a scatter plot for Actual vs. Predicted values
scatter_fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Price', 'y': 'Predicted Price'})
scatter_fig.update_layout(margin=dict(l=0, r=20, t=15, b=0))

# Calculate model performance indicators
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error {mae}; Mean Squared Error {MSE}; R2 score {r2}')
scatter_fig.show()
