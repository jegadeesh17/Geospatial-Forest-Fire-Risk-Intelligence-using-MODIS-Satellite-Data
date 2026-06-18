# %% [markdown]
# # Geospatial Forest Fire Risk Intelligence using MODIS Satellite Data
# 
# 
# 
# Forest fires represent a significant threat to ecosystems and economies, particularly in fire-prone regions across India. This project aims to build an intelligence platform that processes satellite data from NASA's MODIS sensors to identify thermal anomalies and map wildfire risks. By combining geospatial clustering with machine learning, the system provides early warnings and helps in monitoring high-intensity hotspots.
# 
# 
# 
# ### Project Objectives
# 
# - Analyze MODIS satellite data for thermal infrared anomalies.
# 
# - Identify geographic fire zones using spatial clustering (KMeans).
# 
# - Classify high-risk wildfire events using the XGBoost algorithm.
# 
# - Visualize hotspots and regional fire density through interactive mapping.
# 
# - Track seasonal trends to understand fire activity patterns over time.
# 
# 
# 
# ### Dataset Overview
# 
# The project uses the NASA MODIS Active Fire Dataset, covering wildfire observations across India. The data includes thermal infrared bands, fire radiative power (FRP), and detection confidence scores at a spatial resolution of approximately 1km to 4km.
# 
# 
# 
# ### Technical Stack
# 
# - Programming: Python
# 
# - Data Libraries: Pandas, NumPy, Scikit-learn
# 
# - Machine Learning: XGBoost
# 
# - Visualization: Matplotlib, Seaborn, Folium
# 
# 
# 
# ---

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix

# %%
import warnings
warnings.filterwarnings(action="ignore")
%matplotlib inline
pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 1000)

# %%
fires = pd.read_csv(r"C:\Users\jegad\projects\fire_prediction\data\modis_data.csv")    #reading the dataset
fires.head(15)   #show the first 15 instances of dataset


# %%
#show the last 10 instances of dataset
fires.tail(10)

# %%
fires.fillna(fires.median(numeric_only=True), inplace=True)
fires.fillna(fires.mode().iloc[0], inplace=True)

# %% [markdown]
# ### this code splits the date into 3 features day, month and year , and time into hours and minutes

# %%
import pandas as pd

# Load the dataset (assuming the filename is modis_data.csv)
df = pd.read_csv(r"C:\Users\jegad\projects\fire_prediction\data\modis_data.csv")

# 1. Split acq_date into 3 features: day, month, and year
df['acq_date'] = pd.to_datetime(df['acq_date'])
df['year'] = df['acq_date'].dt.year
df['month'] = df['acq_date'].dt.month
df['day'] = df['acq_date'].dt.day

# 2. Split acq_time into hours and minutes
# acq_time is typically in HHMM format (e.g., 1723 becomes 17:23)
# We pad with leading zeros to ensure it's always 4 digits
df['acq_time_str'] = df['acq_time'].astype(int).astype(str).str.zfill(4)
df['hours'] = df['acq_time_str'].str[:2].astype(int)
df['minutes'] = df['acq_time_str'].str[2:].astype(int)

# 3. Create a new column for seasons according to the month
def get_season(month):
    if month in [12, 1, 2]:
        return 1  #winter
    elif month in [3, 4, 5]:
        return  2 #'Spring'  
    elif month in [6, 7, 8]:
        return 3  # 'Summer'
    else:
        return 4 #'Autumn'

df['season'] = df['month'].apply(get_season)

# Display the first few rows to verify the new features
print(df[['acq_date', 'year', 'month', 'day', 'acq_time', 'hours', 'minutes', 'season']].head())


# %% [markdown]
# Sorting the column according to the time and date

# %%
df = df.sort_values(['year', 'month', 'day', 'hours', 'minutes']).reset_index(drop=True)
df.head(5)

# %% [markdown]
# Now we are clustering the regions using lat and long coordinates so that we can find the most fire prone regions

# %%
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=15, random_state=42)

df['region_cluster'] = kmeans.fit_predict(
    df[['latitude', 'longitude']]
)
df.head()

# %% [markdown]
# Now we are going to calculate the

# %% [markdown]
# We are checking how active the cluste is

# %%
df['cluster_fire_count'] = (
    df.groupby('region_cluster')['region_cluster']
    .transform('count')
)

# %% [markdown]
# Monthly fire activity

# %%
df['monthly_cluster_fire_count'] = (
    df.groupby(
        ['region_cluster', 'month']
    )['region_cluster']
    .transform('count')
)

# %% [markdown]
# This is to check how abnormal the hotspot temperature is

# %%
df['brightness_diff'] = (
    df['brightness'] - df['bright_t31']
)
df.head(5)

# %% [markdown]
# The high risk targets

# %%
df['high_risk'] = (
    (df['confidence'] > 80) &
    (df['frp'] > 20)
).astype(int)

# %% [markdown]
# ###  Fire Intensity Categorization
# Segmenting Fire Radiative Power (FRP) into qualitative risk levels: Low, Medium, High, and Extreme.

# %%
df['frp_category'] = pd.cut(
    df['frp'],
    bins=[0, 10, 30, 60, 1000],
    labels=['Low', 'Medium', 'High', 'Extreme']
)

# %% [markdown]
# Encoding day and night to binary

# %%
df['daynight'] = df['daynight'].map({
    'D': 1,
    'N': 0
})

# %% [markdown]
# Encoding the satellite

# %%
df['satellite'] = df['satellite'].map({
    'Terra': 0,
    'Aqua': 1
})

# %%
df.columns.isna()

# %% [markdown]
# Final feature set

# %%
features = [
    'brightness',
    'bright_t31',
    'brightness_diff',
    'scan',
    'track',
    'daynight',
    'month',
    'hours',
    'region_cluster',
    'cluster_fire_count',
    'monthly_cluster_fire_count'
]
target = ['high_risk']

# %% [markdown]
# Train test split

# %%
from sklearn.model_selection import train_test_split

X = df[features]

y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# %% [markdown]
# ## Model training

# %% [markdown]
# ###  Random Forest Classifier
# Training a baseline ensemble model for high-risk wildfire detection.

# %%
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

# %%
from sklearn.metrics import (
    classification_report,
    roc_auc_score
)

print("Classification Report ")
print(classification_report(y_test, rf_pred))

print("ROC-AUC:",
      roc_auc_score(y_test, rf_pred))

# %% [markdown]
# ### XGB

# %%
from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    random_state=42,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

# %%
from sklearn.metrics import (
    classification_report,
    roc_auc_score
)

print("Classification report ")
print(classification_report(y_test, xgb_pred))

print("ROC-AUC:",
      roc_auc_score(y_test, xgb_pred))

# %%


# %% [markdown]
# The XGb is slightly better than RF so we choose this

# %% [markdown]
# ## we get into visualisation and intrepretation

# %%
import folium

fire_map1 = folium.Map(
    location=[22, 80],
    zoom_start=5
)

# %% [markdown]
# adding the fire hotspots

# %%
for _, row in df.sample(5000).iterrows():

    folium.CircleMarker(
        location=[
            row['latitude'],
            row['longitude']
        ],
        radius=3,
        color='red',
        fill=True,
        fill_opacity=0.6,
        popup=f"FRP: {row['frp']}"
    ).add_to(fire_map1)

# %% [markdown]
# ###  Interactive Map Output
# Render the generated Folium map for spatial intelligence analysis.

# %%
fire_map1

# %% [markdown]
# High risk fire map

# %%
import folium

fire_map2 = folium.Map(
    location=[22, 80],
    zoom_start=5
)

# %%
high_risk_df = df[df['high_risk'] == 1]

# %%
for _, row in high_risk_df.sample(5000).iterrows():

    folium.CircleMarker(
        location=[
            row['latitude'],
            row['longitude']
        ],
        radius=max(2, min(row['frp'] / 8, 8)),
        color='darkred' if row['frp'] > 50 else 'orange',
        fill=True,
        fill_opacity=0.6,
        popup=f"""
        FRP: {row['frp']}
        Confidence: {row['confidence']}
        """
    ).add_to(fire_map2)

# %%
fire_map2

# %% [markdown]
# PHASE 6 
# Step 1  Daily Fire Counts

# %%
daily_fire_counts = (
    df.groupby('acq_date')
    .size()
    .reset_index(name='fire_count')
)

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))

plt.plot(
    daily_fire_counts['acq_date'],
    daily_fire_counts['fire_count']
)

plt.title("Daily Fire Trend")

plt.xlabel("Date")

plt.ylabel("Fire Count")

plt.show()

# %% [markdown]
# Step 3  Moving Average Forecast Trend

# %%
daily_fire_counts['moving_avg_7'] = (
    daily_fire_counts['fire_count']
    .rolling(7)
    .mean()
)

# %%
plt.figure(figsize=(12,5))

plt.plot(
    daily_fire_counts['fire_count'],
    label='Actual'
)

plt.plot(
    daily_fire_counts['moving_avg_7'],
    label='7-Day Moving Average'
)

plt.legend()

plt.show()

# %% [markdown]
# ## Summary of Findings
# 
# ### Model Performance
# The XGBoost model achieved 97% accuracy, showing that thermal features like brightness difference and FRP are very effective for identifying high-risk fires. 
# 
# ### Regional Patterns
# The clustering analysis reveals specific areas in India that are consistently hit by seasonal fires. This kind of spatial intelligence is crucial for planning and resource allocation.
# 
# ### Next Steps
# In the future, we could integrate real-time data directly from the NASA FIRMS API and include weather variables like humidity and wind speed to further improve the predictions.


