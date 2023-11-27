from dash import html
from dash import dcc
import joblib
import boto3
from botocore.exceptions import NoCredentialsError

def load_model_from_s3():
    s3 = boto3.client('s3', aws_access_key_id='key_id', aws_secret_access_key='secret_key')
    bucket_name = 'price-ml-model'
    model_key = 'model.joblib'

    try:
        # Download the model file from S3
        s3.download_file(bucket_name, model_key, 'model.joblib')

        # Load the model
        model = joblib.load('model.joblib')
        return model

    except NoCredentialsError:
        print('Credentials not available')


model_instance = load_model_from_s3()

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

num_cols = 3  # Number of columns to divide the categorical features into
categorical_feature_options_split = [
    categorical_feature_options[i::num_cols] for i in range(num_cols)]

layout = html.Div([
    html.H1('Real Estate Price Prediction (Flats)', style={'font-family': 'Arial, sans-serif'}),
    # First row
    html.Div([
        # First column
        html.Div([
            html.H2('Model Performance: Actual vs. Predicted Values',
                    style={'margin-bottom': '10px', 'font-family': 'Arial, sans-serif'}),
            dcc.Graph(id='scatter-plot', figure=model_instance.get_scatter_plot(), style={'margin-top': '10px'})
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
                html.Tr([html.Td('Mean Absolute Error (MAE)'), html.Td(f'{model_instance.get_mae():.2f}')]),
                html.Tr([html.Td('Mean Squared Error (MSE)'), html.Td(f'{model_instance.get_mse():.2f}')]),
                html.Tr([html.Td('R-squared (R2)'), html.Td(f'{model_instance.get_r2():.2f}')])
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
            # Categorical features
            html.H4('Select Categorical Features', style={'margin-top': '20px', 'font-family': 'Arial, sans-serif'}),
            html.Div([
                dcc.Checklist(id=f'checklist-col-{i}', options=options, value=[], style={'margin-bottom': '10px', 'font-family': 'Arial, sans-serif'})
                for i, options in enumerate(categorical_feature_options_split)
            ], style={'columnCount': num_cols}),
            html.Button('Predict Price', id='predict-button', n_clicks=0, style={'margin-top': '20px', 'font-family': 'Arial, sans-serif'}),
            # Displaying prediction
            html.Div(id='predicted-price-output',
                     style={'margin-top': '20px', 'font-size': '18px', 'font-family': 'Arial, sans-serif'})
        ], style={'width': '55%'})
    ], style={'width': '100%', 'margin-top': '20px', 'display': 'inline-block'}),
])
