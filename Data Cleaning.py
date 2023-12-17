import pandas as pd
from haversine import haversine
from sqlalchemy import create_engine

engine = create_engine('postgresql://xxxxxx:yyyyyyy@qqqqqqqqqqq.rds.amazonaws.com:0000/db_name')
sql_query = 'SELECT * FROM uncleaned_flats'
flats_df = pd.read_sql_query(sql_query, engine)

columns_drop = ['Namo numeris:', 'Buto numeris:', 'Unikalus daikto numeris (RC numeris):', 'Aktyvus iki', 'Papildoma įranga:', 'Nuoroda']
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

def unique_val(df, columns):
    for column in columns:
        value_counts = df[column].value_counts()
        print(f'Column: {column}, Value Counts: \n{value_counts}\n')

flats_df['type'] = flats_df['type'].str.replace('Rąstinis', 'Medinis').replace('Karkasinis', 'Kita').replace('Skydinis', 'Kita')
flats_df['mounting'] = flats_df['mounting'].str.replace(' NAUDINGA: Interjero dizaineriai', '')

flats_df['energy_class'] = flats_df['energy_class'].str.strip().fillna('Unspecified')

def redo_energy_class(df, column):
    modified_df = df.copy()
    modified_df['energy_class'] = modified_df[column].apply(lambda x: x if x in ['A', 'A+', 'A++', 'B', 'Not specified'] else 'Lower than B')    
    return modified_df

modified_flats_df = redo_energy_class(flats_df, 'energy_class')

modified_flats_df = pd.get_dummies(modified_flats_df, columns=['type', 'mounting', 'energy_class'], prefix=['type', 'mounting', 'energy_class'], dummy_na=True)

modified_flats_df['heating'] = modified_flats_df['heating'].str.lower().str.strip().str.split(', ')
unique_heating_methods = sorted(modified_flats_df['heating'].explode().unique())

def get_dummies(df, column_name, unique_values):
    for element in unique_values:
        df[element] = 0
        df.loc[df[column_name].apply(lambda x: element in x), element] = 1
get_dummies(modified_flats_df, 'heating', unique_heating_methods)

modified_flats_df['centrinis_sildymas'] = modified_flats_df.apply(lambda row: 1 if row['centrinis'] == 1 or row['centrinis kolektorinis'] == 1 else 0, axis=1)
modified_flats_df.drop(columns=['centrinis', 'centrinis kolektorinis'], inplace=True)

modified_flats_df.rename(columns={'kita': 'sildymas_kita'}, inplace=True)

modified_flats_df['extra_rooms'] = modified_flats_df['extra_rooms'].fillna('none').str.replace('Vieta automobiliui', 'vieta_automobiliui').str.replace('Yra palėpė', 'palepe').str.lower().str.strip().str.split(' ')
unique_extra_rooms = sorted(modified_flats_df['extra_rooms'].explode().unique())
get_dummies(modified_flats_df, 'extra_rooms', unique_extra_rooms)

modified_flats_df['security'] = modified_flats_df['security'].fillna('none').str.replace('Šarvuotos durys', 'Sarvuotos_durys').str.replace('Kodinė laiptinės spyna', 'Kodine_spyna').str.replace('Vaizdo kameros', 'Kameros').str.replace('Budintis sargas', 'Sargas').str.lower().str.strip().str.split(' ')
unique_security = sorted(modified_flats_df['security'].explode().unique())
get_dummies(modified_flats_df, 'security', unique_security)

modified_flats_df['properties'] = modified_flats_df['properties'].str.replace(r'(?<=[a-z\s])(?=[A-Z])', ',', regex=True).fillna('none').str.lower().str.replace(r'^varž.*', 'aukcionas', regex=True).str.split(',').apply(lambda x: [item.strip() for item in x])
unique_properties = sorted(modified_flats_df['properties'].explode().unique())
get_dummies(modified_flats_df, 'properties', unique_properties)

modified_flats_df['building_age'] = 2023 - modified_flats_df['construction_year']
modified_flats_df['building_age_reno'] = 2023 - modified_flats_df['renovation_year']

modified_flats_df['latitude'] = pd.to_numeric(modified_flats_df['latitude'], errors='coerce')
modified_flats_df['longitude'] = pd.to_numeric(modified_flats_df['longitude'], errors='coerce')
modified_flats_df['lat_long'] = list(zip(modified_flats_df['latitude'], modified_flats_df['longitude']))
vilnius_center = (54.68935143850194, 25.270763607406778)
def calculate_distance_to_center(row):
    return haversine(row['lat_long'], vilnius_center)

modified_flats_df['distance_to_center'] = modified_flats_df.apply(calculate_distance_to_center, axis=1)

columns_to_drop = [
    "Namo numeris:", "Buto numeris:", "Unikalus daikto numeris (RC numeris):",
    "Aktyvus iki", "Papildoma įranga:", "extra_rooms", "heating", "energy_class_nan", "type_nan", "mounting_nan",
    "security", "link", "properties", "posted", "edited", "Aktyvus iki", 'latitude', 'longitude', 'lat_long', 'construction_year', 'renovation_year', 'renovuotas namas', 'Reklama:', 'year'
]
modified_flats_df = modified_flats_df.drop(columns=columns_to_drop)

modified_flats_df.dropna(subset=['distance_to_center'], inplace=True)
existing_data_df = pd.read_sql('SELECT * FROM cleaned_flats', con=engine)

# Concatenate existing and new data, dropping duplicates based on all columns except 'price'
combined_df = pd.concat([existing_data_df, modified_flats_df], ignore_index=True)
unique_data_df = combined_df.drop_duplicates(subset=combined_df.columns.difference(['price']), keep='last')
rows_to_insert = len(unique_data_df)
unique_data_df.to_sql('uncleaned_flats', con=engine, if_exists='replace', index=False)
print(f"Number of rows: {rows_to_insert}")
