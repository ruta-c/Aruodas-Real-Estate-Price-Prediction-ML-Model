from dash import html
from dash import dcc, dash_table
import joblib
import boto3
from botocore.exceptions import NoCredentialsError

def load_model_from_s3():
    s3 = boto3.client('s3', aws_access_key_id='key_id', aws_secret_access_key='secret_key')
    bucket_name = 'price-ml-model'
    model_key = 'model.joblib'

    try:
        s3.download_file(bucket_name, model_key, 'model.joblib')
        model = joblib.load('model.joblib')
        return model

    except NoCredentialsError:
        print('Credentials not available')

model_instance = load_model_from_s3()

categorical_feature_options = [
    {'label': 'Concrete Block Construction', 'value': 'type_Blokinis'},
    {'label': 'Energy Class A++', 'value': 'energy_class_A++'},
    {'label': 'Energy Class A', 'value': 'energy_class_A'},
    {'label': 'Aerothermal Heating', 'value': 'aeroterminis'},
    {'label': 'Wooden Construction', 'value': 'type_Medinis'},
    {'label': 'Energy Class A+', 'value': 'energy_class_A+'},
    {'label': 'Gas Heating', 'value': 'dujinis'},
    {'label': 'Monolithic Construction', 'value': 'type_Monolitinis'},
    {'label': 'Energy Class B', 'value': 'energy_class_B'},
    {'label': 'Electric Heating', 'value': 'elektra'},
    {'label': 'Brick Construction', 'value': 'type_Mūrinis'},
    {'label': 'Energy Class Lower Than B', 'value': 'energy_class_Lower than B'},
    {'label': 'Geothermal Heating', 'value': 'geoterminis'},
    {'label': 'Other Construction', 'value': 'type_Kita'},
    {'label': 'Energy Class Not Specified', 'value': 'energy_class_Not specified'},
    {'label': 'Solid Fuel Heating', 'value': 'kietu kuru'},
    {'label': 'Fully Furnished', 'value': 'mounting_Įrengtas'},
    {'label': 'Balcony', 'value': 'balkonas'},
    {'label': 'Central Heating', 'value': 'centrinis_sildymas'},
    {'label': 'Partly Furnished', 'value': 'mounting_Dalinė apdaila'},
    {'label': 'Closet', 'value': 'drabužinė'},
    {'label': 'Solar Power Heating', 'value': 'saulės energija'},
    {'label': 'Not Furnished', 'value': 'mounting_Neįrengtas'},
    {'label': 'Attic', 'value': 'palepe'},
    {'label': 'Liquid Fuel Heating', 'value': 'skystu kuru'},
    {'label': 'Other Furnishing', 'value': 'mounting_Kita'},
    {'label': 'Sauna', 'value': 'pirtis'},
    {'label': 'Other Heating', 'value': 'sildymas_kita'},
    {'label': 'Basement', 'value': 'rūsys'},
    {'label': 'Pantry', 'value': 'sandėliukas'},
    {'label': 'Terrace', 'value': 'terasa'},
    {'label': 'Parking Space', 'value': 'vieta_automobiliui'},
    {'label': 'Security Cameras', 'value': 'kameros'},
    {'label': 'Combination Lock Door', 'value': 'kodine_spyna'},
    {'label': 'Guard', 'value': 'sargas'},
    {'label': 'Steel Doors', 'value': 'sarvuotos_durys'},
    {'label': 'Alarm System', 'value': 'signalizacija'},
    {'label': 'Separate Entrance', 'value': 'atskiras įėjimas'},
    {'label': 'Auction', 'value': 'aukcionas'},
    {'label': 'High Ceilings', 'value': 'aukštos lubos'},
    {'label': 'Attic Apartment', 'value': 'butas palėpėje'},
    {'label': 'Multi-Floor Apartment', 'value': 'butas per kelis aukštus'},
    {'label': 'Part of the Apartment', 'value': 'buto dalis'},
    {'label': 'Internet', 'value': 'internetas'},
    {'label': 'Cable TV', 'value': 'kabelinė televizija'},
    {'label': 'New Electrical Installation', 'value': 'nauja elektros instaliacija'},
    {'label': 'New Plumbing', 'value': 'nauja kanalizacija'},
    {'label': 'Separate Toilet and Bathroom', 'value': 'tualetas ir vonia atskirai'},
    {'label': 'Closed Yard', 'value': 'uždaras kiemas'},
    {'label': 'Kitchen Connected with Room', 'value': 'virtuvė sujungta su kambariu'},
    {'label': 'No Special Properties', 'value': 'none'}
]

options_type = [{'label': option['label'], 'value': option['value']} for option in categorical_feature_options if option['value'] in ['type_Blokinis', 'type_Medinis', 'type_Monolitinis', 'type_Mūrinis', 'type_Kita']]
options_mounting = [{'label': option['label'], 'value': option['value']} for option in categorical_feature_options if option['value'] in ['mounting_Įrengtas', 'mounting_Dalinė apdaila', 'mounting_Neįrengtas', 'mounting_Kita']]
options_energy_class = [{'label': option['label'], 'value': option['value']} for option in categorical_feature_options if option['value'] in ['energy_class_A++', 'energy_class_A+', 'energy_class_A', 'energy_class_B', 'energy_class_Lower than B']]
options_heating = [{'label': option['label'], 'value': option['value']} for option in categorical_feature_options if option['value'] in ['aeroterminis',  'centrinis_sildymas', 'dujinis', 'elektra', 'geoterminis', 'kietu kuru', 'saulės energija', 'skystu kuru', 'sildymas_kita',]]
options_other = [{'label': option['label'], 'value': option['value']} for option in categorical_feature_options if option['value'] in ['balkonas', 'drabužinė', 'none', 'palepe', 'pirtis', 'rūsys', 'sandėliukas', 'terasa', 'vieta_automobiliui', 'kameros', 'kodine_spyna', 'sargas', 'sarvuotos_durys', 'signalizacija', 'atskiras įėjimas', 'aukcionas', 'aukštos lubos', 'butas palėpėje', 'butas per kelis aukštus', 'buto dalis', 'internetas', 'kabelinė televizija', 'nauja elektros instaliacija', 'nauja kanalizacija', 'tualetas ir vonia atskirai', 'uždaras kiemas', 'virtuvė sujungta su kambariu']]
options_other_sorted = sorted(options_other, key=lambda x: x['label'])

input_style = {'margin-bottom': '10px', 'width': '100%', 'padding': '10px', 'box-sizing': 'border-box',
               'font-family': 'Arial, sans-serif', 'border': '1px solid #ccc', 'font-size': '16px', 'color': '#333333'}
dropdown_style = {'margin-bottom': '10px', 'font-family': 'Arial, sans-serif', 'color': '#333333', 'width': '100%'}
button_style = {'margin-top': '20px', 'width': '95%', 'height': '40px', 'font-family': 'Arial, sans-serif',
                'border': '1px solid #ccc', 'font-size': '16px', 'background-color': '#ffffff', 'color': '#333333'}

tab1_content = html.Div([
html.Div([
        # First column
        html.Div([
            html.H2('Model Performance: Actual vs. Predicted Values',
                    style={'margin-bottom': '10px', 'font-family': 'Arial, sans-serif'}),
            dcc.Graph(id='scatter-plot', figure=model_instance.get_scatter_plot(), style={'margin-top': '10px'})
        ], style={'display': 'inline-block', 'width': '50%', 'vertical-align': 'top'}),
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
            html.Tr('2023-12-17', style={'font-family': 'Arial, sans-serif'})
        ], style={'display': 'inline-block', 'width': '50%', 'vertical-align': 'top'})
    ])
])

tab2_content = html.Div([
    html.Div([
        html.H2('Feature Selection and Price Prediction', style={
            'margin-bottom': '10px',
            'padding': '10px',
            'box-sizing': 'border-box',
            'font-family': 'Arial, sans-serif'}),

        html.Div([
            html.Div([
                html.Label('Area, sqm', style={'font-size': '16px', 'font-family': 'Arial, sans-serif'}),
                dcc.Input(id='area', type='number', placeholder='29', value=29, style=input_style),
            ], style={'width': '45%', 'display': 'inline-block', 'margin-right': '5%'}),

            html.Div([
                html.Label('Number of Rooms', style={'font-size': '16px', 'font-family': 'Arial, sans-serif'}),
                dcc.Input(id='rooms', type='number', placeholder='1', value=1, style=input_style),
            ], style={'width': '45%', 'display': 'inline-block'}),
        ]),

        html.Div([
            html.Div([
                html.Label('Floor', style={'font-size': '16px', 'font-family': 'Arial, sans-serif'}),
                dcc.Input(id='floor', type='number', placeholder='2', value=2, style=input_style),
            ], style={'width': '45%', 'display': 'inline-block', 'margin-right': '5%'}),

            html.Div([
                html.Label('Total Number of Floors', style={'font-size': '16px', 'font-family': 'Arial, sans-serif'}),
                dcc.Input(id='floors', type='number', placeholder='5', value=5, style=input_style),
            ], style={'width': '45%', 'display': 'inline-block'}),
        ]),

        html.Div([
            html.Div([
                html.Label('Building Age, years', style={'font-size': '16px', 'font-family': 'Arial, sans-serif'}),
                dcc.Input(id='building_age', type='number', placeholder='39', value=39, style=input_style),
            ], style={'width': '45%', 'display': 'inline-block', 'margin-right': '5%'}),

            html.Div([
                html.Label('Renovated Building Age, years (the same as Building age if not renovated)',
                           style={'font-size': '16px', 'font-family': 'Arial, sans-serif'}),
                dcc.Input(id='building_age_reno', type='number', placeholder='8', value=8, style=input_style),
            ], style={'width': '45%', 'display': 'inline-block'}),
        ]),

        html.Div([
            html.Div([
                html.Label('Distance to Center, km', style={'font-size': '16px', 'font-family': 'Arial, sans-serif'}),
                dcc.Input(id='distance_to_center', type='number', placeholder='2.3', value=2.3, style=input_style),
            ], style={'width': '45%', 'display': 'inline-block', 'margin-right': '5%', 'vertical-align': 'top'}),

            html.Div([
                html.Label('Type of Building', style={'font-size': '16px', 'font-family': 'Arial, sans-serif'}),
                dcc.Dropdown(id='dropdown-type', options=options_type, multi=False, value='type_Blokinis',
                             style=dropdown_style),
            ], style={'width': '45%', 'display': 'inline-block', 'margin-right': '5%', 'vertical-align': 'top'}),
        ]),

        html.Div([
            html.Label('Mounting', style={'font-size': '16px', 'font-family': 'Arial, sans-serif'}),
            dcc.Dropdown(id='dropdown-mounting', options=options_mounting, multi=False,
                         value='mounting_Įrengtas', style=dropdown_style),
        ], style={'width': '45%', 'display': 'inline-block', 'margin-right': '5%', 'vertical-align': 'top'}),

        html.Div([
            html.Label('Energy Class', style={'font-size': '16px', 'font-family': 'Arial, sans-serif'}),
            dcc.Dropdown(id='dropdown-energy-class', options=options_energy_class, multi=False,
                         value='energy_class_A', style=dropdown_style),
        ], style={'width': '45%', 'display': 'inline-block', 'margin-right': '5%', 'vertical-align': 'top'}),

        html.Div([
            html.Label('Heating Type', style={'font-size': '16px', 'font-family': 'Arial, sans-serif'}),
            dcc.Dropdown(id='dropdown-heating', options=options_heating, multi=False,
                         value='centrinis_sildymas', style=dropdown_style),
        ], style={'width': '45%', 'display': 'inline-block', 'margin-right': '5%', 'vertical-align': 'top'}),

        html.Div([
            html.Label('Other properties', style={'font-size': '16px', 'font-family': 'Arial, sans-serif'}),
            dcc.Dropdown(id='dropdown-other', options=options_other_sorted, multi=True, value=['none'],
                         style=dropdown_style),
        ], style={'width': '45%', 'display': 'inline-block', 'vertical-align': 'top'}),

        html.Button('Predict Price', id='predict-button', n_clicks=0, style=button_style),
    ], style={'width': '50%', 'float': 'left'}),

    html.Div(id='price-prediction-section', children=[
        html.H2('Price prediction and similar flats', style={
            'margin-bottom': '10px',
            'padding': '10px',
            'box-sizing': 'border-box',
            'font-family': 'Arial, sans-serif'}),
        html.Div(id='predicted-price-output',
                 style={'margin-bottom': '20px', 'font-size': '18px', 'font-family': 'Arial, sans-serif'}),

        dcc.Tabs([
            dcc.Tab(
                label='Similar Flat No. 1',
                style={'font-family': 'Arial, sans-serif'},
                selected_style={'font-family': 'Arial, sans-serif', 'border': '1px solid #d6d6d6'},
                children=[
                    dash_table.DataTable(
                        id='similar-flats-table-1',
                        columns=[
                            {'name': 'Property', 'id': 'Property'},
                            {'name': 'Value', 'id': 'Value'},
                        ],
                        style_table={'width': '100%', 'font-family': 'Arial, sans-serif'},
                        style_cell={'textAlign': 'left', 'font-family': 'Arial, sans-serif'},
                        style_data={'whiteSpace': 'normal', 'font-family': 'Arial, sans-serif'},
                    ),
                ]
            ),
            dcc.Tab(
                label='Similar Flat No. 2',
                style={'font-family': 'Arial, sans-serif'},
                selected_style={'font-family': 'Arial, sans-serif', 'border': '1px solid #d6d6d6'},
                children=[
                    dash_table.DataTable(
                        id='similar-flats-table-2',
                        columns=[
                            {'name': 'Property', 'id': 'Property'},
                            {'name': 'Value', 'id': 'Value'},
                        ],
                        style_table={'width': '100%', 'font-family': 'Arial, sans-serif'},
                        style_cell={'textAlign': 'left', 'font-family': 'Arial, sans-serif'},
                        style_data={'whiteSpace': 'normal', 'font-family': 'Arial, sans-serif'},
                    ),
                ]
            ),
            dcc.Tab(
                label='Similar Flat No. 3',
                style={'font-family': 'Arial, sans-serif'},
                selected_style={'font-family': 'Arial, sans-serif', 'border': '1px solid #d6d6d6'},
                children=[
                    dash_table.DataTable(
                        id='similar-flats-table-3',
                        columns=[
                            {'name': 'Property', 'id': 'Property'},
                            {'name': 'Value', 'id': 'Value'},
                        ],
                        style_table={'width': '100%', 'font-family': 'Arial, sans-serif'},
                        style_cell={'textAlign': 'left', 'font-family': 'Arial, sans-serif'},
                        style_data={'whiteSpace': 'normal', 'font-family': 'Arial, sans-serif'},
                    ),
                ]
            ),
        ]),
    ], style={'width': '50%', 'float': 'right'}),
], style={'overflow': 'hidden'})

layout = html.Div([
    html.H1('Real Estate Price Prediction (Flats)', style={'font-family': 'Arial, sans-serif'}),

    dcc.Tabs(id='tabs', value='tab1', style={'font-family': 'Arial, sans-serif'}, children=[
        dcc.Tab(label='Model Performance', children=[tab1_content], id='tab1'),
        dcc.Tab(label='Feature Selection & Prediction', children=[tab2_content], id='tab2'),
    ]),

    html.Div(id='tabs-content')
])
