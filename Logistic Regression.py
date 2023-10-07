import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
)

# Constants
DATA_FILE = "flats.csv"
THRESHOLD = 0.446 #Tested as best performing

# Data Loading and Preprocessing
flats_df = pd.read_csv(DATA_FILE)
columns_to_drop = [
    "Namo numeris:", "Buto numeris:", "Unikalus daikto numeris (RC numeris):",
    "Aktyvus iki", "Papildoma įranga:", "Unnamed: 0", "extra_rooms", "heating",
    "security", "link", "properties", "posted", "edited", "Aktyvus iki"
]
flats_df = flats_df.drop(columns=columns_to_drop).dropna()

# Visualize Data
plt.hist(flats_df["price_sqm_log"], bins=1000)
plt.title("Price Distribution")
plt.xlabel("Price per Square Meter (Log Scale)")
plt.ylabel("Frequency")
plt.show()

# Create Price Categories
flats_df["price_cat_exp"] = (flats_df["price_sqm_log"] > 8.096398).astype(int)

# Model Training and Evaluation
columns_to_use = ['area', 'rooms', 'floor', 'floors', 'looked_by',
                  'saved', 'type_Blokinis', 'type_Kita','type_Medinis', 
                  'type_Monolitinis', 'type_Mūrinis', 'type_nan',
                  'mounting_Dalinė apdaila', 'mounting_Kita', 'mounting_Neįrengtas',
                  'mounting_Įrengtas', 'mounting_nan', 'energy_class_A',
                  'energy_class_A+', 'energy_class_A++', 'energy_class_B',
                  'energy_class_C', 'energy_class_D', 'energy_class_F', 'energy_class_G',       
                  'energy_class_nan', 'aeroterminis', 'centrinis',       
                  'centrinis kolektorinis', 'dujinis', 'elektra', 'geoterminis',       
                  'kietu kuru', 'kita', 'saulės energija', 'skystu kuru', 'balkonas',       
                  'drabužinė', 'none', 'palepe', 'pirtis', 'rūsys', 'sandėliukas',       
                  'terasa', 'vieta_automobiliui', 'kameros', 'kodine_spyna', 'sargas',       
                  'sarvuotos_durys', 'signalizacija', 'atskiras įėjimas', 'aukštos lubos',       
                  'butas palėpėje', 'butas per kelis aukštus', 'buto dalis', 'internetas',       
                  'kabelinė televizija', 'nauja elektros instaliacija',       
                  'nauja kanalizacija', 'renovuotas namas', 'tualetas ir vonia atskirai',       
                  'uždaras kiemas', 'virtuvė sujungta su kambariu',       
                  'building_age', 'building_age_reno', 'distance_to_center'
]

X = flats_df[columns_to_use]
y = flats_df["price_cat_exp"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(random_state=42, max_iter=6000)
model.fit(X_train, y_train)

def custom_predict(model, X, threshold):
    probs = model.predict_proba(X) 
    return (probs[:, 1] > threshold).astype(int)

y_pred = custom_predict(model, X_test, threshold=THRESHOLD)

# Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nConfusion Matrix:")
print(confusion)
print("\nClassification Report:")
print(classification_rep)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
