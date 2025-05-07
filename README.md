# Anomaly Detection in Spatial-Temporal Traffic Data

This project identifies and analyzes **anomalies** in **traffic accident profiles** using **spatial-temporal data analysis** techniques, integrating data pre-processing and statistical analysis.  
The analysis mainly focuses on the **effects of lockdown events** on accident patterns across Brazilian states.


---

## 🗂 Project Structure

| File | Description |
|:---|:---|
| `main_AnomalyDetection.py` | Main script: loads data, processes it, detects anomalies, and generates reports and plots. |
| `PreProcessing.py` | Cleans and translates accident datasets. Standardizes dates and groups accident causes. |
| `TimeSeriesTransformation.py` | Handles time series creation, resampling, decomposition, and visualization. |
| `TimeSeriesAutoCorrelation.py` | Calculates spatial autocorrelation scores (Local and Global Moran's Index) and timing scores. |
| `Analysis.py` | Performs spatial analysis, validates results, and detects profile changes through regression models. |
| `Utils.py` | Utility functions (e.g., table export to LaTeX+PDF, DataFrame intersection). |
| `Dictionaries.py` | Contains predefined mappings: accident cause categories, fleet sizes, and neighbor states. |

---

## 📋 Main Features

- **Preprocessing:**
  - Normalize date formats.
  - Translate accident cause names into broader English categories.
  - Reduce and clean datasets for analysis.

- **Time Series Handling:**
  - Create state-based time series for accident counts.
  - Resample by days, weeks, or months.
  - STL decomposition into trend, seasonality, and residuals.

- **Spatial Analysis:**
  - Calculate spatial correlation scores:
    - **Timing correlation**
    - **Local Moran Index (LMI)**
    - **Global Moran Index (GMI)**
  - Validate spatial autocorrelation against random models.

- **Anomaly Detection:**
  - Perform **regression analysis** comparing pre-lockdown and during-lockdown profiles.
  - Detect anomalies based on residuals and slopes.
  - Classify anomalies into phenomena like "Intensification", "Attenuation", or "Flip".

- **Result Presentation:**
  - Export tables automatically into **PDFs** using LaTeX.
  - Generate exploratory plots (Moran scatter plots, neighborhood comparisons, decomposition graphs).

---

## 🛠 How to Run

1. **Prepare your data:**
   - A CSV file (`PreProc_Data.csv`) containing accident data with date and location information.
   - A shapefile (`BR_UF_2022.shp`) for mapping.

2. **Run the Main Script:**
   ```bash
   python main_AnomalyDetection.py
   ```

3. **Outputs:**
   - Tables saved as PDFs inside `/Tables/`.
   - Plots saved inside folders like `/SpatialAnalysis/`, `/Maps_QuartilScore/`, etc.
   - Console printouts highlight the profile changes and anomalies.

---

## 📛 Requirements

- Python 3.7+
- Required packages:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scipy`
  - `statsmodels`
  - `scikit-learn'

Install Python libraries with:

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn
```

---

## ⚙ Configuration

In `main_AnomalyDetection.py`, you can change:

| Parameter | Purpose | Default |
|:---|:---|:---|
| `csv_file_name` | Path to your cleaned CSV data. | `'PreProc_Data.csv'` |
| `shp_file_name` | Path to the shapefile for state boundaries. | `'BR_UF_2022/BR_UF_2022.shp'` |
| `granularity` | Number of days/weeks/months to group. | `1` |
| `count` | Resampling basis (`'days'` or `'months'`). | `'months'` |
| `alpha` | Strength of the timing mapping function. | `2` |

---

## 📄 License

Feel free to modify and adapt it to your dataset!


