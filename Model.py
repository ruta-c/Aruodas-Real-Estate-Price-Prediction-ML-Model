import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
from sqlalchemy import create_engine

class RealEstateModel:
    def __init__(self):
        self._load_data()
        self._set_features()
        self._train_model()

    def _load_data(self):
        engine = create_engine('postgresql://xxxxxx:yyyyyyy@qqqqqqqqqqq.rds.amazonaws.com:0000/db_name')
        sql_query = 'SELECT * FROM cleaned_flats'
        self.df = pd.read_sql_query(sql_query, engine)

    def _set_features(self):
        self.features_list = ['area', 'rooms', 'floor', 'floors',
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
        self.exclude = ['area', 'distance_to_center']
        self.df[self.features_list] = self.df[self.features_list].apply(lambda col: col.astype('int32') if col.name not in self.exclude else col)
        return self.features_list

    def _train_model(self):
        X = self.df[self.features_list]
        y = self.df['price_sqm']
        X_train, X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = RandomForestRegressor(max_depth=13, min_samples_split=2, n_estimators=739, random_state=42)
        self.model.fit(X_train, y_train)
        self.y_pred = self.model.predict(X_test)
        return self.model

    def get_mae(self):
        mae = mean_absolute_error(self.y_test, self.y_pred)
        return mae

    def get_mse(self):
        mse = mean_squared_error(self.y_test, self.y_pred)
        return mse

    def get_r2(self):
        r2 = r2_score(self.y_test, self.y_pred)
        return r2

    def get_scatter_plot(self):
        scatter_fig = px.scatter(x=self.y_test, y=self.y_pred, labels={'x': 'Actual Price', 'y': 'Predicted Price'})
        scatter_fig.update_layout(margin=dict(l=0, r=20, t=15, b=0))
        return scatter_fig

    def get_training_data(self):
        return self.df[self.features_list], self.df['price_sqm']

    def predict_price(self, input_data):
        return self.model.predict(input_data)
