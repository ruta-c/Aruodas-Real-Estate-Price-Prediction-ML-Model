import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
)

class RealEstateClassifier:
    def __init__(self, train_data_file, validation_data_file, threshold=0.446):
        self.train_data = pd.read_csv(train_data_file)
        self.validation_data = pd.read_csv(validation_data_file)
        self.threshold = threshold
        self.columns_to_drop = [
            'Namo numeris:',
            'Buto numeris:',
            'Unikalus daikto numeris (RC numeris):',
            'Aktyvus iki',
            'Papildoma įranga:',
            'Unnamed: 0',
            'extra_rooms',
            'heating',
            'security',
            'link',
            'properties',
            'posted',
            'edited',
            'Aktyvus iki',
            
        ]
        self.columns_to_use = [
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
            'energy_class_D',
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
            'skystu kuru',
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
            'buto dalis',
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
        self.prepare_data()

    def prepare_data(self):
        self.train_data = self.train_data.drop(columns=self.columns_to_drop).dropna()
        self.train_data['price_cat_exp'] = (self.train_data['price_sqm_log'] > 8.096398).astype(int)

        self.validation_data = self.validation_data.drop(columns=self.columns_to_drop).dropna()
        self.validation_data['price_cat_exp'] = (self.validation_data['price_sqm_log'] > 8.096398).astype(int)

    def train_model(self):
        X_train = self.train_data[self.columns_to_use]
        y_train = self.train_data['price_cat_exp']
        X_validation = self.validation_data[self.columns_to_use]
        y_validation = self.validation_data['price_cat_exp']

        self.model = LogisticRegression(random_state=42, max_iter=6000)
        self.model.fit(X_train, y_train)

        y_pred_probs = self.model.predict_proba(X_validation)[:, 1] > self.threshold
        self.y_pred = y_pred_probs.astype(int)

        self.accuracy = accuracy_score(y_validation, self.y_pred)
        self.confusion = confusion_matrix(y_validation, self.y_pred)
        self.classification_rep = classification_report(y_validation, self.y_pred)
        self.fpr, self.tpr, _ = roc_curve(y_validation, self.y_pred)
        self.roc_auc = roc_auc_score(y_validation, self.y_pred)

    def plot_roc_curve(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.fpr, self.tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {self.roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()
        pass



if __name__ == '__main__':
    train_data_file = 'flats.csv'  # Path to the training data CSV file
    validation_data_file = 'validation_flats_.csv'  # Path to the validation data CSV file

    classifier = RealEstateClassifier(train_data_file, validation_data_file)
    classifier.train_model()

    print('Accuracy:', classifier.accuracy)
    print('\nConfusion Matrix:')
    print(classifier.confusion)
    print('\nClassification Report:')
    print(classifier.classification_rep)

    classifier.plot_roc_curve()
