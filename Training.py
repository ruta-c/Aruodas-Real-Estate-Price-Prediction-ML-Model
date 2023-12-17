from Model import RealEstateModel
import joblib

model = RealEstateModel()
X_train, y_train = model.get_training_data()
joblib.dump((X_train, y_train), 'training_data.joblib')
scatter_plot = model.get_scatter_plot()
scatter_json = scatter_plot.to_json()
joblib.dump(model, 'model.joblib')
