from dash import Dash, dcc, html, Input, Output
from sklearn.model_selection import train_test_split
from sklearn import linear_model, tree, neighbors
from sklearn import metrics, datasets
import plotly.express as px
import pandas as pd

app = Dash(__name__)

MODELS = {'Logistic Regression': linear_model.LogisticRegression,
          'Decision Tree': tree.DecisionTreeClassifier,
          'k-NN': neighbors.KNeighborsClassifier}

app.layout = html.Div([
    html.H4("Analysis of the ML model's results using ROC and PR curves"),
    html.P("Select model:"),
    dcc.Dropdown(
        id='dropdown',
        options=[
            {'label': 'Logistic Regression', 'value': 'Logistic Regression'},
            {'label': 'Decision Tree', 'value': 'Decision Tree'},
            {'label': 'k-NN', 'value': 'k-NN'}
    ],
    value='Logistic Regression',
    clearable=False
    ),
    dcc.Graph(id="graph"),
])


@app.callback(
    Output("graph", "figure"), 
    [Input('dropdown', "value")])
def train_and_display(model_name):
    data = pd.read_csv('flats.csv')
    data = data.dropna()
    features = [
            'area',
            'rooms',
            'floor',
            'floors',
            'looked_by',
            'saved',
            'type_Blokinis',
            'type_Kita',
            'type_Medinis', 
            'type_Monolitinis',
            'type_Mūrinis',
            'type_nan',
            'mounting_Dalinė apdaila',
            'mounting_Kita',
            'mounting_Neįrengtas',
            'mounting_Įrengtas',
            'mounting_nan',
            'energy_class_A',
            'energy_class_A+',
            'energy_class_A++',
            'energy_class_B',
            'energy_class_C',
            'energy_class_F',
            'energy_class_G',
            'energy_class_nan',
            'aeroterminis',
            'centrinis',       
            'centrinis kolektorinis',
            'dujinis',
            'elektra',
            'geoterminis',
            'kietu kuru',
            'kita',
            'saulės energija',
            'balkonas',
            'drabužinė',
            'none',
            'palepe',
            'pirtis',
            'rūsys',
            'sandėliukas',
            'terasa',
            'vieta_automobiliui',
            'kameros',
            'kodine_spyna',
            'sargas',
            'sarvuotos_durys',
            'signalizacija',
            'atskiras įėjimas',
            'aukštos lubos',
            'butas palėpėje',
            'butas per kelis aukštus',
            'internetas',
            'kabelinė televizija',
            'nauja elektros instaliacija',
            'nauja kanalizacija',
            'renovuotas namas',
            'tualetas ir vonia atskirai',
            'uždaras kiemas',
            'virtuvė sujungta su kambariu',
            'building_age',
            'building_age_reno',
            'distance_to_center'
        ]

    X = data[features]  # Features
    y = data['price_cat_exp']  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = MODELS[model_name]()
    model.fit(X_train, y_train)

    y_score = model.predict_proba(X_test)[:, 1] > 0.446

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)
    score = metrics.auc(fpr, tpr)

    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={score:.4f})',
        labels=dict(
            x='False Positive Rate', 
            y='True Positive Rate'))
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1)
    return fig


app.run_server(debug=True)


