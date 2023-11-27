import dash
from dash import Output, Input, State
import pandas as pd
from Layout import layout, num_cols
import joblib
import boto3
from botocore.exceptions import NoCredentialsError

app = dash.Dash(__name__)
server = app.server


def load_model_from_s3():
    s3 = boto3.client('s3', aws_access_key_id='key_id', aws_secret_access_key='secret_key)
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

app.layout = layout

correct_feature_order = ['area', 'rooms', 'floor', 'floors',
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

@app.callback(
    Output('predicted-price-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [
        State('area', 'value'),
        State('rooms', 'value'),
        State('floor', 'value'),
        State('floors', 'value'),
        State('building_age', 'value'),
        State('building_age_reno', 'value'),
        State('distance_to_center', 'value'),
        *[State(f'checklist-col-{i}', 'value') for i in range(num_cols)]
    ]
)
def update_predicted_price(n_clicks, area, rooms, floor, floors, building_age, building_age_reno, distance_to_center,
                           *selected_features):
    # Check if the button was clicked
    if n_clicks and n_clicks > 0:
        if None not in [area, rooms, floor, floors, building_age, building_age_reno, distance_to_center]:
            selected_features_flat = [item for sublist in selected_features for item in sublist]
            selected_features_set = set(selected_features_flat)
            selected_features_dict = {feature: 1 if feature in selected_features_set else 0 for feature in
                                      correct_feature_order}

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
            input_data = selected_features_dict.copy()  
            input_data.update(numeric_data)  
            input_df = pd.DataFrame([input_data])
            input_df = input_df[correct_feature_order]

            # Use the pretrained model to predict the price
            predicted_price = model_instance.predict_price(input_df)[0]

            return f'Predicted Price: {predicted_price:.2f} €/sqm'

        else:
            return "Please fill out all input fields."
    else:
        # If the button was not clicked, return an empty string
        return ''

#port = int(os.environ.get("PORT", 8050))
if __name__ == '__main__':
    app.run_server(debug=True)
