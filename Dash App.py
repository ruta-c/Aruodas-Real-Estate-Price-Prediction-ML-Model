from sqlalchemy import create_engine
import dash
from dash import dcc, html, Output, Input
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

engine = create_engine('postgresql://postgres:PASSWORD@localhost:0000/name')
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

# Split the dataset
X = df[features]
y = df['price_sqm']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the RandomForestRegressor model
model = RandomForestRegressor(min_samples_split=2, n_estimators=739, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Create a scatter plot for Actual vs. Predicted values
scatter_fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Price', 'y': 'Predicted Price'})
scatter_fig.update_layout(margin=dict(l=0, r=20, t=15, b=0))
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Initialize the Dash app
app = dash.Dash(__name__)

categorical_feature_options = [
    {'label': 'Concrete Block', 'value':'type_Blokinis'},
    {'label': 'Energy Class A++', 'value': 'energy_class_A++'},
    {'label': 'Aerothermal Heating', 'value': 'aeroterminis'},
    {'label': 'Wooden', 'value': 'type_Medinis'},
    {'label': 'Energy Class A+', 'value': 'energy_class_A+'},
    {'label': 'Gas Heating', 'value': 'dujinis'},
    {'label': 'Monolithic', 'value': 'type_Monolitinis'},
    {'label': 'Energy Class B', 'value': 'energy_class_B'},
    {'label': 'Electric Heating', 'value': 'elektra'},
    {'label': 'Bricks', 'value': 'type_Mūrinis'},
    {'label': 'Energy Class Lower Than B', 'value': 'energy_class_Lower than B'},
    {'label': 'Geothermal Heating', 'value': 'geoterminis'},
    {'label': 'Other type of material', 'value':'type_Kita'},
    {'label': 'Energy Class Not specified', 'value': 'energy_class_Not specified'},
    {'label': 'Solid Fuel Heating', 'value': 'kietu kuru'},
    {'label': 'Fully mounted', 'value': 'mounting_Įrengtas'},
    {'label': 'Balcony', 'value': 'balkonas'},
    {'label': 'Central Heating', 'value': 'centrinis_sildymas'},
    {'label': 'Partly Mounted', 'value': 'mounting_Dalinė apdaila'},
    {'label': 'Closet', 'value': 'drabužinė'},
    {'label': 'Solar Power Heating', 'value': 'saulės energija'},
    {'label': 'Not Mounted', 'value': 'mounting_Neįrengtas'},
    {'label': 'Attic', 'value': 'palepe'},
    {'label': 'Liquid Fuel Heating', 'value': 'skystu kuru'},
    {'label': 'Mounted: Other', 'value': 'mounting_Kita'},
    {'label': 'Sauna', 'value': 'pirtis'},
    {'label': 'Other Heating', 'value': 'sildymas_kita'},
    {'label': 'Basement', 'value': 'rūsys'},
    {'label': 'Pantry', 'value': 'sandėliukas'},
    {'label': 'Terrace', 'value': 'terasa'},
    {'label': 'Parking Spot', 'value': 'vieta_automobiliui'},
    {'label': 'Cameras', 'value': 'kameros'},
    {'label': 'Combination Lock Door', 'value': 'kodine_spyna'},
    {'label': 'Guard', 'value': 'sargas'},
    {'label': 'Steel Door', 'value': 'sarvuotos_durys'},
    {'label': 'Alarm', 'value': 'signalizacija'},
    {'label': 'Separate Entrance', 'value': 'atskiras įėjimas'},
    {'label': 'Auction', 'value': 'aukcionas'},
    {'label': 'High Ceiling', 'value': 'aukštos lubos'},
    {'label': 'Flat in the Attic', 'value': 'butas palėpėje'},
    {'label': 'Flat over Several Floors', 'value': 'butas per kelis aukštus'},
    {'label': 'Part of the Flat', 'value': 'buto dalis'},
    {'label': 'Internet', 'value': 'internetas'},
    {'label': 'Cable TV', 'value': 'kabelinė televizija'},
    {'label': 'New Electrical Installation', 'value': 'nauja elektros instaliacija'},
    {'label': 'New Plumbing', 'value': 'nauja kanalizacija'},
    {'label': 'Toilet and Bathroom Separated', 'value': 'tualetas ir vonia atskirai'},
    {'label': 'Closed Yard', 'value': 'uždaras kiemas'},
    {'label': 'Kitchen Connected with Room', 'value': 'virtuvė sujungta su kambariu'},
    {'label': 'No Special Properties', 'value': 'none'}
]

# Define the categorical feature options
num_cols = 3  # Number of columns to divide the categorical features into
categorical_feature_options_split = [
    categorical_feature_options[i::num_cols] for i in range(num_cols)
]

app.layout = html.Div([
    html.H1('Real Estate Price Prediction (Flats)', style={'font-family': 'Arial, sans-serif'}),

    # First row
    html.Div([
        # First column
        html.Div([
            html.H2('Model Performance: Actual vs. Predicted Values',
                    style={'margin-bottom': '10px', 'font-family': 'Arial, sans-serif'}),
            dcc.Graph(id='scatter-plot', figure=scatter_fig, style={'margin-top': '10px'})
        ], style={'display': 'inline-block', 'width': '55%', 'vertical-align': 'top'}),

        # Second column
        html.Div([
            html.H2('About', style={'font-family': 'Arial, sans-serif'}),
            html.Tr('This machine learning project is focused on predicting flat prices per square meter in Vilnius. The project utilizes data collected from the real estate advertisements website aruodas.lt through web scraping. The chosen machine learning model, RandomForestRegressor from the Scikit-Learn library, was selected for its superior performance.', style={'font-family': 'Arial, sans-serif', 'text-align': 'justify'}),
            html.H2('Model Performance Indicators', style={'font-family': 'Arial, sans-serif'}),
            html.Table([
                html.Tr([
                    html.Th('Metric', style={'text-align': 'left', 'border-bottom': '1px solid #ddd',
                                             'font-family': 'Arial, sans-serif'}),
                    html.Th('Value', style={'text-align': 'left', 'border-bottom': '1px solid #ddd',
                                            'font-family': 'Arial, sans-serif'})
                ]),
                html.Tr([html.Td('Mean Absolute Error (MAE)'), html.Td(f'{mae:.2f}')]),
                html.Tr([html.Td('Mean Squared Error (MSE)'), html.Td(f'{mse:.2f}')]),
                html.Tr([html.Td('R-squared (R2)'), html.Td(f'{r2:.2f}')])
            ], style={'width': '100%', 'margin-top': '5px', 'font-family': 'Arial, sans-serif'}),

            html.H2('Training data renewed:', style={'font-family': 'Arial, sans-serif'}),
            html.Tr('2023-10-19', style={'font-family': 'Arial, sans-serif'})
        ], style={'display': 'inline-block', 'width': '40%', 'vertical-align': 'top'})
    ]),

    # Second row
    html.Div([
        html.Div([
            html.H2('Feature Selection and Price Prediction', style={'font-family': 'Arial, sans-serif'}),
            dcc.Input(id='area', type='number', placeholder='Area', style={'margin-bottom': '10px'}),
            dcc.Input(id='rooms', type='number', placeholder='Rooms', style={'margin-bottom': '10px'}),
            dcc.Input(id='floor', type='number', placeholder='Floor', style={'margin-bottom': '10px'}),
            dcc.Input(id='floors', type='number', placeholder='Total floors', style={'margin-bottom': '10px'}),
            dcc.Input(id='building_age', type='number', placeholder='Building age',
                      style={'margin-bottom': '10px'}),
            dcc.Input(id='building_age_reno', type='number', placeholder='Building age (renovated)',
                      style={'margin-bottom': '10px'}),
            dcc.Input(id='distance_to_center', type='number', placeholder='Distance to center',
                      style={'margin-bottom': '10px'}),

            # Categorical features divided into multiple columns
            html.H4('Select Categorical Features', style={'margin-top': '20px', 'font-family': 'Arial, sans-serif'}),
            html.Div([
                dcc.Checklist(id=f'checklist-col-{i}', options=options, value=[], style={'margin-bottom': '10px', 'font-family': 'Arial, sans-serif'})
                for i, options in enumerate(categorical_feature_options_split)
            ], style={'columnCount': num_cols}),

            html.Button('Predict Price', id='predict-button', n_clicks=0, style={'margin-top': '20px', 'font-family': 'Arial, sans-serif'}),

            # Display the predicted price here
            html.Div(id='predicted-price-output',
                     style={'margin-top': '20px', 'font-size': '18px', 'font-family': 'Arial, sans-serif'})
        ], style={'width': '55%'})
    ], style={'width': '100%', 'margin-top': '20px', 'display': 'inline-block'}),
])

# Correct feature order based on the model's training data
correct_feature_order = features

@app.callback(
    Output('predicted-price-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [
        Input('area', 'value'),
        Input('rooms', 'value'),
        Input('floor', 'value'),
        Input('floors', 'value'),
        Input('building_age', 'value'),
        Input('building_age_reno', 'value'),
        Input('distance_to_center', 'value'),
        *[Input(f'checklist-col-{i}', 'value') for i in range(num_cols)]
    ]
)
def update_predicted_price(n_clicks, area, rooms, floor, floors, building_age, building_age_reno, distance_to_center,
                           *selected_features):
    # Flatten the list of selected features
    selected_features_flat = [item for sublist in selected_features for item in sublist]
    # Convert the flattened list to a set
    selected_features_set = set(selected_features_flat)
    selected_features_dict = {feature: 1 if feature in selected_features_set else 0 for feature in
                              correct_feature_order}
    # Check if the button was clicked
    if n_clicks > 0:
        # Check if all required fields are filled
        if None not in [area, rooms, floor, floors, building_age, building_age_reno, distance_to_center]:
            # Create a dictionary to hold input values and selected categorical features
            numeric_data = {
                'area': area,
                'rooms': rooms,
                'floor': floor,
                'floors': floors,
                'building_age': building_age,
                'building_age_reno': building_age_reno,
                'distance_to_center': distance_to_center
            }

            # Update the input_data dictionary with selected_features_dict
            input_data = selected_features_dict.copy()  # Create a copy of selected_features_dict
            input_data.update(numeric_data)  # Update the copy with numeric_data
            input_df = pd.DataFrame([input_data])

            # Ensure input data has the correct feature order
            input_df = input_df[correct_feature_order]

            predicted_price = model.predict(input_df)[0]
            return f'Predicted Price: {predicted_price:.2f} €/sqm'

        else:
            return "Please fill out all input fields."
    else:
        # If the button was not clicked, return an empty string
        return ''

if __name__ == '__main__':
    app.run_server(debug=True)
