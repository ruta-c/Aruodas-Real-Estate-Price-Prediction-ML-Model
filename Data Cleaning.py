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

flats_df["heating"] = flats_df["heating"].str.lower().str.strip().str.split(', ')
unique_heating_methods = sorted(flats_df["heating"].explode().unique())

def get_dummies(df, column_name, unique_values):
    for element in unique_values:
        df[element] = 0
        df.loc[df[column_name].apply(lambda x: element in x), element] = 1
get_dummies(flats_df, "heating", unique_heating_methods)

flats_df["extra_rooms"] = flats_df["extra_rooms"].fillna("none").str.replace("Vieta automobiliui", "vieta_automobiliui").str.replace("Yra palėpė", "palepe").str.lower().str.strip().str.split(' ')
unique_extra_rooms = sorted(flats_df["extra_rooms"].explode().unique())
get_dummies(flats_df, "extra_rooms", unique_extra_rooms)

flats_df["security"] = flats_df["security"].fillna("none").str.replace("Šarvuotos durys", "Sarvuotos_durys").str.replace("Kodinė laiptinės spyna", "Kodine_spyna").str.replace("Vaizdo kameros", "Kameros").str.replace("Budintis sargas", "Sargas").str.lower().str.strip().str.split(' ')
unique_security = sorted(flats_df["security"].explode().unique())
get_dummies(flats_df, "security", unique_security)

flats_df["properties"] = flats_df["properties"].str.replace(r'(?<=[a-z\s])(?=[A-Z])', ',', regex=True).fillna("none").str.lower().str.split(',').apply(lambda x: [item.strip() for item in x])
unique_properties = sorted(flats_df["properties"].explode().unique())
get_dummies(flats_df, "properties", unique_properties)

print(flats_df.head(10))
print(flats_df.dtypes)
flats_df.to_csv("C:/Users/rceid/OneDrive - Lietuvos sveikatos mokslu universitetas/Documents/kvailioju programuoju/Aruodas-Real-Estate-Price-Prediction-ML-Model/flats.csv")
