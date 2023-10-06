import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


flats_df = pd.read_csv('flats.csv')
flats_df['price_sqm'] = flats_df['price'] / flats_df['area']
#print(flats_df['price_sqm'].describe())

# Remove outliers (e.g., values greater than 10000)
filtered_df = flats_df[flats_df['price_sqm'] <= 10000]
filtered_df.hist(column='price_sqm', bins=30)

plt.show()

print(filtered_df['price_sqm'].describe())
