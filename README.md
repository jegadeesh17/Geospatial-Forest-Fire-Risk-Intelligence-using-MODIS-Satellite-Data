# Geospatial Forest Fire Risk Intelligence using MODIS Satellite Data

---

### **Project Overview**

Forest fires pose severe environmental, ecological, and economic threats worldwide. This project leverages NASA MODIS satellite fire detection data, geospatial analytics, and machine learning to build an AI-powered wildfire risk intelligence platform focused on India's fire-prone regions.

The system analyzes thermal anomalies, spatial wildfire patterns, and temporal fire activity to identify and classify high-risk wildfire events. By integrating machine learning with geospatial visualization, the platform enables early risk assessment and hotspot monitoring for disaster management and environmental agencies.

---

### **Key Features**

* **Satellite Data Intelligence:** Processes MODIS wildfire detection data using thermal and radiative features.
* **Geospatial Risk Analytics:** Identifies wildfire-prone zones using KMeans spatial clustering.
* **Machine Learning Classification:** Uses XGBoost to classify high-risk wildfire events.
* **Thermal Anomaly Detection:** Engineers brightness-based anomaly features for improved fire risk detection.
* **Interactive Fire Mapping:** Visualizes wildfire hotspots and intensity using Folium heatmaps.
* **Temporal Trend Analysis:** Tracks seasonal and monthly wildfire activity patterns.
* **Interactive Streamlit Dashboard:** Live deployment for real-time monitoring and analytics.
* **Contextual Cluster Insights:** AI-generated interpretation of spatial clusters for better stakeholder decision-making.

---

### **Dataset**

* **Source:** NASA MODIS Active Fire Dataset
* **Coverage:** Pan-India wildfire observations
* **Spatial Resolution:** Approx. 1km–4km satellite fire detection grids

#### **Key Features**

* Brightness temperature
* Thermal infrared bands (`bright_t31`)
* Fire Radiative Power (`frp`)
* Fire confidence score
* Day/Night satellite observations
* Latitude and longitude coordinates
* Acquisition timestamps

---

### **Project Structure**

```bash
Forest-Fire-Risk-Analytics/
│
├── data/                     # MODIS wildfire datasets
│
├── notebooks/                # Jupyter notebooks for EDA and modeling
│
├── models/                   # Saved trained models
│
├── visualizations/           # Heatmaps and output plots
│
├── src/
│   ├── preprocessing.py      # Data cleaning
│   ├── feature_engineering.py # Feature engineering logic
│   ├── modeling.py           # Model training pipeline
│   ├── forecasting.py        # Forecasting logic
│   ├── evaluation.py         # Model evaluation
│   ├── insights.py           # AI-generated insights and cluster interpretation
│   ├── visualization.py      # Geospatial visualization scripts
│   └── app.py                # Streamlit dashboard application
│
├── fire_detection.ipynb      # Main end-to-end notebook
├── requirements.txt          # Python dependencies
└── README.md
```

---

### **How It Works**

### **1. Data Preprocessing & Feature Engineering**

* Cleans MODIS wildfire observations and handles missing values.
* Extracts temporal features: Month, Hour, and Seasonal activity.
* Creates advanced wildfire intelligence features:

| Feature                      | Purpose                                    |
| ---------------------------- | ------------------------------------------ |
| `brightness_diff`            | Detects thermal anomalies                  |
| `region_cluster`             | Identifies wildfire-prone geographic zones |
| `cluster_fire_count`         | Measures regional wildfire activity        |
| `monthly_cluster_fire_count` | Captures seasonal hotspot density          |

---

### **2. Spatial Clustering**

Uses KMeans clustering on latitude and longitude coordinates to identify geographically similar wildfire regions.

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=15, random_state=42)
df['region_cluster'] = kmeans.fit_predict(df[['latitude', 'longitude']])
```

---

### **3. High-Risk Fire Classification**

The system engineers a custom wildfire risk label using Fire Radiative Power (`frp`) and detection confidence score, then classifies high-risk wildfire events using an XGBoost Classifier.

---

### **Model Performance**

| Metric                     | Score |
| -------------------------- | ----- |
| Accuracy                   | 97%   |
| ROC-AUC                    | 0.94  |
| F1-Score (High-Risk Class) | 0.90  |

---

### **Interactive Application Deployment**

The project features an interactive **Streamlit Web Application** designed with clean UI aesthetics, enabling users to explore geospatial clusters, map active hot spots, and monitor risk profiles.

#### **To Launch the Platform Locally:**
```powershell
python -m streamlit run ".\forest fire risk intelligence\src\app.py"
```

---

### **Technology Stack**

| Category             | Tools                 |
| -------------------- | --------------------- |
| Programming          | Python                |
| Data Processing      | Pandas, NumPy         |
| Machine Learning     | Scikit-learn, XGBoost |
| Visualization        | Matplotlib, Seaborn   |
| Geospatial Analytics | Folium                |
| Notebook Environment | Jupyter Notebook      |
| Web Framework        | Streamlit             |

---

### **Getting Started**

### **1. Clone Repository**

```bash
git clone https://github.com/yourusername/Forest-Fire-Risk-Analytics.git

cd Forest-Fire-Risk-Analytics
```

---

### **2. Install Dependencies**

```bash
pip install -r requirements.txt
```

---

### **3. Launch Notebook**

```bash
jupyter notebook
```

Open:

```bash
fire_detection.ipynb
```

---

### **4. Train the Model**

```python
from xgboost import XGBClassifier

xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
```

---

### **Example Use Case**

A disaster management authority can use the platform to:

1. Monitor high-risk wildfire regions in real time
2. Identify seasonal wildfire hotspots
3. Prioritize aerial surveillance in high-risk zones
4. Improve emergency response planning using spatial fire intelligence

---

### **Future Improvements**

* Real-time NASA FIRMS API integration
* Weather and humidity data integration
* Deep learning-based fire spread forecasting
* Burn severity estimation using satellite imagery

---

### **Contributors**

* **Jegadeesh D** — Data preprocessing, feature engineering, machine learning, geospatial analytics, and visualization

---

### **License**

MIT License
