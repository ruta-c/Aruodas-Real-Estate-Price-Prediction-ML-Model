from Model import RealEstateModel
import joblib

model = RealEstateModel()
scatter_plot = model.get_scatter_plot()

scatter_json = scatter_plot.to_json()

# Use joblib for model serialization
joblib.dump(model, 'model.joblib')
