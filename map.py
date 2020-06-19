import geopandas as gdp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import map US state map data from US Census website...I downloaded the data first and imported from my computer
us_state_map = gdp.read_file('C:/Users/huffm/Desktop/MSDS 692/Flu Shot Prediction/cb_2019_us_all_20m/cb_2019_us_state_20m/cb_2019_us_state_20m.shp')
us_state_map.plot()

# Convert 'NAME' column to uppercase to match CDC data
us_state_map['NAME'] = us_state_map['NAME'].str.upper()
print(us_state_map)
# Drop Puerto Rico, Hawaii, and Alaska to keep the graph to the continental US in order to keep the map more compact
us_state_map = us_state_map.drop(us_state_map.index[[1,21,50]])

# Import CDC data, remove NA values and calculate ratio of H1N1 vaccines in each state
map_features = pd.read_csv('C:/Users/huffm/Desktop/MSDS 692/Flu Shot Prediction/Data from CDC/cdc_data.csv', usecols=['VACC_H1N1_F', 'STATE'])

print(map_features.head())
print(map_features.isnull().sum())
map_features = map_features.dropna()
print(map_features.isnull().sum())

def ratio(x):
    n_1 = sum(x['VACC_H1N1_F'].values == 'Yes')
    n_0 = sum(x['VACC_H1N1_F'].values == 'No')
    return n_1/n_0

by_state = map_features.groupby('STATE').apply(ratio)

print(by_state.head())
by_state = by_state.to_frame()
print(by_state)
by_state.rename(columns={0: 'ratio'}, inplace=True)

#by_state.set_index('STATE')
## Merge Geo dataset with state vaccine info
merged = pd.merge(us_state_map, by_state, left_on='NAME', right_on='STATE')
print(merged.head())

## Create Map from Merged Dataset
var = 'ratio'
vmin, vmax = merged['ratio'].min(), merged['ratio'].max()

# create figure and axes for Matplotlib
fig, ax = plt.subplots(1, figsize=(10, 5))
merged.plot(column=var, cmap='Greens', linewidth=0.8, ax=ax, edgecolor='0.8')
ax.axis('off')
ax.set_title('Ratio of Individuals Vaccinated for H1N1', fontname='Times New Roman')
# Add legend
sm = plt.cm.ScalarMappable(cmap='Greens', norm=plt.Normalize(vmin=merged['ratio'].min(), vmax=merged['ratio'].max()))
sm._A = []
cbar = fig.colorbar(sm)
plt.show()