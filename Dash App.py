import dash
from dash import Output, Input, State, html
import pandas as pd
import joblib
import boto3
from botocore.exceptions import NoCredentialsError
from sklearn.neighbors import NearestNeighbors
from Layout import layout, tab1_content, tab2_content

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server


def load_model_from_s3(model_key):
    s3 = boto3.client('s3', aws_access_key_id='key_id', aws_secret_access_key='secret_key')
    bucket_name = 'price-ml-model'
  
    try:
        s3.download_file(bucket_name, model_key, model_key)
        model = joblib.load(model_key)
        return model

    except NoCredentialsError:
        print('Credentials not available')

model_instance = load_model_from_s3('model.joblib')
(X_train, y_train) = load_model_from_s3('training_data.joblib')

app.layout = html.Div([
    html.Div(id='selected-flat-info-store', style={'display': 'none'}),
    layout
])

correct_feature_order = [
    'area', 'rooms', 'floor', 'floors',
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
    'distance_to_center'
]

nn_model = NearestNeighbors(n_neighbors=4)
nn_model.fit(X_train[correct_feature_order])

def process_dropdown_values(feature, selected_value, input_data):
    if selected_value:
        if feature == 'other':
            for value in selected_value:
                column_name = f'{value}'
                input_data[column_name] = 1
                print(f"Feature: {feature}, Selected Value: {value}, Column Name: {column_name}")
        else:
            column_name = f'{selected_value}'
            input_data[column_name] = 1
            print(f"Feature: {feature}, Selected Value: {selected_value}, Column Name: {column_name}")

def create_input_data(area, rooms, floor, floors, building_age, building_age_reno, distance_to_center,
                      dropdown_type, dropdown_mounting, dropdown_energy_class, dropdown_heating, dropdown_other):
    input_data = {
        'area': area,
        'rooms': rooms,
        'floor': floor,
        'floors': floors,
        'building_age': building_age,
        'building_age_reno': building_age_reno,
        'distance_to_center': distance_to_center,
    }

    dropdowns = {
        'type': dropdown_type,
        'mounting': dropdown_mounting,
        'energy_class': dropdown_energy_class,
        'heating': dropdown_heating,
        'other': dropdown_other
    }

    for feature, selected_value in dropdowns.items():
        process_dropdown_values(feature, selected_value, input_data)

    return pd.DataFrame([input_data]).reindex(columns=correct_feature_order, fill_value=0)

def update_similar_flats_df(indices, input_data, label_mapping):
    similar_flats_df = pd.DataFrame(columns=['Property', 'Value'])

    for i, idx in enumerate(indices[0][1:]):
        similar_flats_df.loc[len(similar_flats_df)] = {
            'Property': f"Similar Flat {i + 1} - Predicted Price",
            'Value': f"{y_train.iloc[idx]:.2f} €/sqm"
        }

        for col in ['area', 'rooms', 'floor', 'floors', 'building_age', 'building_age_reno', 'distance_to_center']:
            label = label_mapping.get(col, col)

            if col == 'distance_to_center':
                value = f"{X_train[col].iloc[idx]:.2f} km"
            else:
                value = X_train[col].iloc[idx]

            similar_flats_df.loc[len(similar_flats_df)] = {'Property': label, 'Value': value}

        for col in X_train.columns:
            if col != 'rooms' and X_train[col].iloc[idx] == 1:
                label_display = label_mapping.get(col, col)
                similar_flats_df.loc[len(similar_flats_df)] = {'Property': label_display, 'Value': 'Yes'}

    return similar_flats_df

@app.callback(
    Output('tabs-content', 'children'),
    [Input('tabs', 'value')]
)
def update_tab_content(selected_tab):
    if selected_tab is None:
        selected_tab = 'tab1'
    if selected_tab == 'tab1':
        return tab1_content
    elif selected_tab == 'tab2':
        return tab2_content

@app.callback(
    Output('price-prediction-section', 'style'),
    [Input('predict-button', 'n_clicks')]
)
def toggle_price_prediction_section(n_clicks):
    if n_clicks and n_clicks > 0:
        return {'display': 'block'}
    else:
        return {'display': 'none'}

@app.callback(
    [
        Output('predicted-price-output', 'children'),
        Output('similar-flats-table-1', 'data'),
        Output('similar-flats-table-2', 'data'),
        Output('similar-flats-table-3', 'data'),
    ],
    [Input('predict-button', 'n_clicks')],
    [
        State('area', 'value'),
        State('rooms', 'value'),
        State('floor', 'value'),
        State('floors', 'value'),
        State('building_age', 'value'),
        State('building_age_reno', 'value'),
        State('distance_to_center', 'value'),
        State('dropdown-type', 'value'),
        State('dropdown-mounting', 'value'),
        State('dropdown-energy-class', 'value'),
        State('dropdown-heating', 'value'),
        State('dropdown-other', 'value'),
    ]
)
def update_predicted_price(n_clicks, area, rooms, floor, floors, building_age, building_age_reno, distance_to_center,
                           dropdown_type, dropdown_mounting, dropdown_energy_class, dropdown_heating, dropdown_other):

    if n_clicks and n_clicks > 0:
        if all(v is not None and v >= 0 for v in [area, rooms, floor, floors, building_age, building_age_reno, distance_to_center]):
            if floor <= floors and building_age >= building_age_reno:
                input_data = create_input_data(area, rooms, floor, floors, building_age, building_age_reno, distance_to_center,
                                               dropdown_type, dropdown_mounting, dropdown_energy_class, dropdown_heating, dropdown_other)

                predicted_price = model_instance.predict_price(input_data)[0]
                _, indices = nn_model.kneighbors(input_data)

                label_mapping = {
                    'area': 'Area (sqm)',
                    'rooms': 'Number of Rooms',
                    'floor': 'Floor',
                    'floors': 'Number of Floors',
                    'building_age': 'Building Age (years)',
                    'building_age_reno': 'Building Renovation Age (years)',
                    'distance_to_center': 'Distance to Center (km)',
                    'type_Blokinis': 'Concrete Block Construction',
                    'energy_class_A++': 'Energy Class A++',
                    'energy_class_A': 'Energy Class A',
                    'aeroterminis': 'Aerothermal Heating',
                    'type_Medinis': 'Wooden Construction',
                    'energy_class_A+': 'Energy Class A+',
                    'dujinis': 'Gas Heating',
                    'type_Monolitinis': 'Monolithic Construction',
                    'energy_class_B': 'Energy Class B',
                    'elektra': 'Electric Heating',
                    'type_Mūrinis': 'Brick Construction',
                    'energy_class_Lower than B': 'Energy Class Lower Than B',
                    'geoterminis': 'Geothermal Heating',
                    'type_Kita': 'Other Construction',
                    'energy_class_Not specified': 'Energy Class Not Specified',
                    'kietu kuru': 'Solid Fuel Heating',
                    'mounting_Įrengtas': 'Fully Furnished',
                    'balkonas': 'Balcony',
                    'centrinis_sildymas': 'Central Heating',
                    'mounting_Dalinė apdaila': 'Partly Furnished',
                    'drabužinė': 'Closet',
                    'saulės energija': 'Solar Power Heating',
                    'mounting_Neįrengtas': 'Not Furnished',
                    'palepe': 'Attic',
                    'skystu kuru': 'Liquid Fuel Heating',
                    'mounting_Kita': 'Other Furnishing',
                    'pirtis': 'Sauna',
                    'sildymas_kita': 'Other Heating',
                    'rūsys': 'Basement',
                    'sandėliukas': 'Pantry',
                    'terasa': 'Terrace',
                    'vieta_automobiliui': 'Parking Space',
                    'kameros': 'Security Cameras',
                    'kodine_spyna': 'Combination Lock Door',
                    'sargas': 'Security Guard',
                    'sarvuotos_durys': 'Steel Doors',
                    'signalizacija': 'Alarm System',
                    'atskiras įėjimas': 'Separate Entrance',
                    'aukcionas': 'Auction',
                    'aukštos lubos': 'High Ceilings',
                    'butas palėpėje': 'Attic Apartment',
                    'butas per kelis aukštus': 'Multi-Floor Apartment',
                    'buto dalis': 'Part of the Apartment',
                    'internetas': 'Internet Connection',
                    'kabelinė televizija': 'Cable TV',
                    'nauja elektros instaliacija': 'New Electrical Installation',
                    'nauja kanalizacija': 'New Plumbing',
                    'tualetas ir vonia atskirai': 'Separate Toilet and Bathroom',
                    'uždaras kiemas': 'Closed Courtyard',
                    'virtuvė sujungta su kambariu': 'Kitchen Connected with Room',
                    'none': 'No Special Properties'
                }
                predicted_price_output = "Predicted Price: {:.2f} €/sqm".format(predicted_price)

                similar_flats_dfs = [pd.DataFrame(columns=['Property', 'Value']) for _ in range(3)]

                for i, idx in enumerate(indices[0][1:4]):  # Consider only the first three similar flats
                    similar_flats_dfs[i].loc[len(similar_flats_dfs[i])] = {
                        'Property': f"Price (€/sqm)",
                        'Value': f"{y_train.iloc[idx]:.2f}"
                    }

                    for col in ['area', 'rooms', 'floor', 'floors', 'building_age', 'building_age_reno', 'distance_to_center']:
                        label = label_mapping.get(col, col)

                        if col == 'distance_to_center':
                            value = f"{X_train[col].iloc[idx]:.2f}"
                        else:
                            value = X_train[col].iloc[idx]

                        similar_flats_dfs[i].loc[len(similar_flats_dfs[i])] = {'Property': label, 'Value': value}

                    for col in X_train.columns:
                        if col != 'rooms' and X_train[col].iloc[idx] == 1:
                            label_display = label_mapping.get(col, col)
                            similar_flats_dfs[i].loc[len(similar_flats_dfs[i])] = {'Property': label_display, 'Value': 'Yes'}

                return predicted_price_output, similar_flats_dfs[0].to_dict('records'), similar_flats_dfs[1].to_dict('records'), similar_flats_dfs[2].to_dict('records')
            else:
                return "Please fill out all input fields. Floor should be less than or equal to total number of floors, and building age should be greater (or equal) than renovated building age.", [], [], []
        else:
            return "Error: Please fill out all input fields with positive values.", [], [], []
    else:
        return '', [], [], []

if __name__ == '__main__':
    app.run_server(debug=True)
