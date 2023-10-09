import pandas as pd
import numpy as np
from haversine import haversine
from sqlalchemy import create_engine

engine = create_engine('postgresql://postgres:odievai1@localhost:5432/aruodas')
sql_query = 'SELECT * FROM flats'
flats_df = pd.read_sql_query(sql_query, engine)
engine.dispose()

print(flats_df.head())

#Drop unnecessary columns
columns_drop = ['Namo numeris:', 'Buto numeris:', 'Unikalus daikto numeris (RC numeris):', 'Aktyvus iki', 'Papildoma įranga:']
flats_df.drop(columns=columns_drop)

#Rename columns
columns_rename = {
    'Price (EUR)': 'price',
    'Plotas:': 'area',
    'Kambarių sk.:': 'rooms',
    'Aukštas:': 'floor',
    'Aukštų sk.:': 'floors',
    'Metai:': 'year',
    'Pastato tipas:': 'type',
    'Šildymas:': 'heating',
    'Įrengimas:': 'mounting',
    'Pastato energijos suvartojimo klasė:': 'energy_class',
    'Apsauga:': 'security',
    'Nuoroda': 'link',
    'Įdėtas': 'posted',
    'Redaguotas': 'edited',
    'Peržiūrėjo': 'looked_by',
    'Latitude': 'latitude',
    'Longitude': 'longitude',
    'Ypatybės:': 'properties',
    'Įsiminė': 'saved',
    'Papildomos patalpos:': 'extra_rooms'
}
flats_df.rename(columns=columns_rename, inplace=True)
flats_df.drop_duplicates()

#Correct data types of the columns
flats_df['price'] = flats_df['price'].str.replace('[^0-9]', '', regex=True).astype(int) # Without regex=True not working
flats_df['area'] = flats_df['area'].str.replace(',', '.').replace('[^0-9.]', '', regex=True).astype(float)
flats_df[['construction_year', 'renovation_year']] = flats_df['year'].str.split(',', expand=True)
flats_df['construction_year'] = flats_df['construction_year'].str.replace('[^0-9]', '', regex=True).astype(int)
flats_df['renovation_year'] = flats_df['renovation_year'].str.replace('[^0-9]', '', regex=True).fillna(flats_df['construction_year']).astype(int)
flats_df['looked_by'] = flats_df['looked_by'].str.extract(r'(\d+)(?=\/)').fillna('0').astype(int)
def convert_to_int(df, columns):
    for column in columns:
        df[column] = df[column].fillna('0').astype(int)
convert_to_int(flats_df, ['rooms', 'floor', 'floors', 'saved'])

#Additional column 
flats_df['price_sqm'] = flats_df['price'] / flats_df['area'] 

#Handle categorical columns
flats_df[['type', 'mounting', 'energy_class']].describe()
def unique_val(df, columns):
    for column in columns:
        value_counts = df[column].value_counts()
        print(f'Column: {column}, Value Counts: \n{value_counts}\n')

unique_val(flats_df, ['type', 'mounting', 'energy_class'])

flats_df['type'] = flats_df['type'].str.replace('Rąstinis', 'Medinis').replace('Karkasinis', 'Kita').replace('Skydinis', 'Kita')
flats_df['mounting'] = flats_df['mounting'].str.replace(' NAUDINGA: Interjero dizaineriai', '')
flats_df = pd.get_dummies(flats_df, columns=['type', 'mounting', 'energy_class'], prefix=['type', 'mounting', 'energy_class'], dummy_na=True)

#Handle categorical columns

flats_df['heating'] = flats_df['heating'].str.lower().str.strip().str.split(', ')
unique_heating_methods = sorted(flats_df['heating'].explode().unique())

def get_dummies(df, column_name, unique_values):
    for element in unique_values:
        df[element] = 0
        df.loc[df[column_name].apply(lambda x: element in x), element] = 1
get_dummies(flats_df, 'heating', unique_heating_methods)

flats_df['extra_rooms'] = flats_df['extra_rooms'].fillna('none').str.replace('Vieta automobiliui', 'vieta_automobiliui').str.replace('Yra palėpė', 'palepe').str.lower().str.strip().str.split(' ')
unique_extra_rooms = sorted(flats_df['extra_rooms'].explode().unique())
get_dummies(flats_df, 'extra_rooms', unique_extra_rooms)

flats_df['security'] = flats_df['security'].fillna('none').str.replace('Šarvuotos durys', 'Sarvuotos_durys').str.replace('Kodinė laiptinės spyna', 'Kodine_spyna').str.replace('Vaizdo kameros', 'Kameros').str.replace('Budintis sargas', 'Sargas').str.lower().str.strip().str.split(' ')
unique_security = sorted(flats_df['security'].explode().unique())
get_dummies(flats_df, 'security', unique_security)

flats_df['properties'] = flats_df['properties'].str.replace(r'(?<=[a-z\s])(?=[A-Z])', ',', regex=True).fillna('none').str.lower().str.replace(r'^varž.*', 'aukcionas', regex=True).str.split(',').apply(lambda x: [item.strip() for item in x])
unique_properties = sorted(flats_df['properties'].explode().unique())
get_dummies(flats_df, 'properties', unique_properties)

# Remove extreme outliers
flats_df = flats_df[(flats_df['price_sqm'] >= 525) & (flats_df['price_sqm'] <= 9100)].copy()

# Normalizing using log
flats_df['price_sqm_log'] = np.log(flats_df['price_sqm'])

# Contruction/renovation year modification into age
flats_df['building_age'] = 2023 - flats_df['construction_year']
flats_df['building_age_reno'] = 2023 - flats_df['renovation_year']

# Using latitude and longitude to calculate distance to Vilnius center (54.68935143850194, 25.270763607406778)
flats_df['latitude'] = pd.to_numeric(flats_df['latitude'], errors='coerce')
flats_df['longitude'] = pd.to_numeric(flats_df['longitude'], errors='coerce')
flats_df['lat_long'] = list(zip(flats_df['latitude'], flats_df['longitude']))
vilnius_center = (54.68935143850194, 25.270763607406778)
def calculate_distance_to_center(row):
    return haversine(row['lat_long'], vilnius_center)

flats_df['distance_to_center'] = flats_df.apply(calculate_distance_to_center, axis=1)
#flats_df = flats_df.dropna()

flats_df["price_cat_exp"] = (flats_df["price_sqm_log"] > 8.096398).astype(int)
columns_to_drop = [
    "Namo numeris:", "Buto numeris:", "Unikalus daikto numeris (RC numeris):",
    "Aktyvus iki", "Papildoma įranga:", "extra_rooms", "heating",
    "security", "link", "properties", "posted", "edited", "Aktyvus iki"
]
flats_df = flats_df.drop(columns=columns_to_drop)
print(flats_df.dtypes)
print(flats_df.columns)
print(flats_df.head())

flats_df.to_csv('C:/Users/rceid/OneDrive - Lietuvos sveikatos mokslu universitetas/Documents/kvailioju programuoju/Aruodas-Real-Estate-Price-Prediction-ML-Model/flats.csv')
