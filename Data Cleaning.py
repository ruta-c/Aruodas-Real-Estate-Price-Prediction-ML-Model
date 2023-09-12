import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql://postgres:odievai1@localhost:5432/aruodas")
sql_query = "SELECT * FROM flats"
flats_df = pd.read_sql_query(sql_query, engine)
engine.dispose()

print(flats_df.head())

#Drop unnecessary columns
columns_drop = ["Namo numeris:", "Buto numeris:", "Unikalus daikto numeris (RC numeris):", "Aktyvus iki", "Papildoma įranga:"]
flats_df.drop(columns=columns_drop)

#Rename columns
columns_rename = {
    "Price (EUR)": "price",
    "Plotas:": "area",
    "Kambarių sk.:": "rooms",
    "Aukštas:": "floor",
    "Aukštų sk.:": "floors",
    "Metai:": "year",
    "Pastato tipas:": "type",
    "Šildymas:": "heating",
    "Įrengimas:": "mounting",
    "Pastato energijos suvartojimo klasė:": "energy_class",
    "Apsauga:": "security",
    "Nuoroda": "link",
    "Įdėtas": "posted",
    "Redaguotas": "edited",
    "Peržiūrėjo": "looked_by",
    "Latitude": "latitude",
    "Longitude": "longitude",
    "Ypatybės:": "properties",
    "Įsiminė": "saved",
    "Papildomos patalpos:": "extra_rooms"
}
flats_df.rename(columns=columns_rename, inplace=True)

#Correct data types of the columns
flats_df["price"] = flats_df["price"].str.replace("[^0-9]", "", regex=True).astype(int) # Without regex=True not working
flats_df["area"] = flats_df["area"].str.replace(",", ".").replace("[^0-9.]", "", regex=True).astype(float)
flats_df[["construction_year", "renovation_year"]] = flats_df["year"].str.split(",", expand=True)
flats_df["construction_year"] = flats_df["construction_year"].str.replace("[^0-9]", "", regex=True).astype(int)
flats_df["renovation_year"] = flats_df["renovation_year"].str.replace("[^0-9]", "", regex=True).fillna(flats_df["construction_year"]).astype(int)
flats_df["looked_by"] = flats_df["looked_by"].str.extract(r"(\d+)(?=\/)").fillna("0").astype(int)
def convert_to_int(df, columns):
    for column in columns:
        df[column] = df[column].fillna("0").astype(int)
convert_to_int(flats_df, ["rooms", "floor", "floors", "saved"]) 

#Handle categorical columns
flats_df[["type", "mounting", "energy_class"]].describe()
def unique_val(df, columns):
    for column in columns:
        value_counts = df[column].value_counts()
        print(f"Column: {column}, Value Counts: \n{value_counts}\n")

unique_val(flats_df, ["type", "mounting", "energy_class"])

flats_df["type"] = flats_df["type"].str.replace("Rąstinis", "Medinis").replace("Karkasinis", "Kita").replace("Skydinis", "Kita")
flats_df["mounting"] = flats_df["mounting"].str.replace(" NAUDINGA: Interjero dizaineriai", "")
flats_df = pd.get_dummies(flats_df, columns=["type", "mounting", "energy_class"], prefix=["type", "mounting", "energy_class"], dummy_na=True)

#Handle categorical columns

flats_df["heating"] = flats_df["heating"].str.lower().str.split(",")
unique_heating_methods = sorted(set([method.strip() for methods in flats_df["heating"] for method in methods]))
#ENDED HERE
for method in unique_heating_methods:
    flats_df[method] = flats_df["heating"].apply(lambda x: 1 if method in x else 0)
flats_df.drop(columns=["heating"], inplace=True)

flats_df["extra_rooms"] = flats_df["extra_rooms"].fillna("")
unique_extra_rooms = set()
for methods in flats_df["extra_rooms"]:
    if methods:
        unique_extra_rooms.update(methods.split(' '))
unique_extra_rooms = {method.strip() for method in unique_extra_rooms}
for method in unique_extra_rooms:
    column_name = f"extra_room_{method}"
    flats_df[column_name] = flats_df["extra_rooms"].apply(lambda x: 1 if method in x.split(' ') else 0)
flats_df.drop(columns=["extra_rooms"], inplace=True)

flats_df["security"] = flats_df["security"].str.replace("Šarvuotos durys", "Sarvuotos_durys").str.replace("Kodinė laiptinės spyna", "Kodine_spyna").str.replace("Vaizdo kameros", "Kameros").str.replace("Budintis sargas", "Sargas")
flats_df["security"] = flats_df["security"].fillna("")
unique_security = set()
for methods in flats_df["security"]:
    if methods:
        unique_security.update(methods.split(' '))
unique_security = {method.strip() for method in unique_security}
for method in unique_security:
    column_name = f"security_{method}"
    flats_df[column_name] = flats_df["security"].apply(lambda x: 1 if method in x.split(' ') else 0)
flats_df.drop(columns=["security"], inplace=True)

flats_df["properties"] = flats_df["properties"].str.replace(r'(?<=[a-z\s])(?=[A-Z])', ',', regex=True)
flats_df["properties"] = flats_df["properties"].fillna("")
flats_df["properties"] = flats_df["properties"].str.split(',')
unique_properties = set()
for properties_list in flats_df["properties"]:
    unique_properties.update(properties_list)
unique_properties = {method.strip() for method in unique_properties}
for method in unique_properties:
    column_name = f"properties_{method}"
    flats_df[column_name] = flats_df["properties"].apply(lambda x: 1 if method in x else 0)
flats_df.drop(columns=["properties"], inplace=True)

print(flats_df.head(10))
print(flats_df.dtypes)
flats_df.to_csv("C:/Users/rceid/OneDrive - Lietuvos sveikatos mokslu universitetas/Documents/kvailioju programuoju/Aruodas-Real-Estate-Price-Prediction-ML-Model/flats.csv")
