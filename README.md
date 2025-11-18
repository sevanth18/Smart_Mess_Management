
# 📘 Smart Hostel Mess Management System

Forecasting Attendance & Optimizing Food Portions using Data Analytics
A data-driven solution to help hostel messes predict attendance, reduce food waste, and plan portions using forecasting, statistics, and interactive dashboards.

---

## 🚀 Features

- **Meal-wise Attendance Forecasting** (Holt-Winters Exponential Smoothing)
- **Two-Way ANOVA** to analyze impact of Day & Meal Type
- **Portion Optimization** based on waste per plate data
- **Interactive Visualizations** (attendance, waste trends, and correlations)
- **Streamlit Web App** for real-time dashboards & planning
- **Food Waste Reduction** (aim: 15–20%) driven by analytics

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:**
  - `Pandas`, `NumPy` – Data handling
  - `Statsmodels` – Forecasting & ANOVA
  - `Matplotlib`, `Seaborn` – Data visualizations
  - `Streamlit` – Dashboard/Web App UI
  - `Scikit-learn` – Preprocessing, ML pipeline

---

## 📂 Project Structure

```
Smart_Mess_Management/
│
├── Food_Waste_data.csv         # Food wastage per meal data
├── Mess_Attendance_Data.csv    # Raw attendance data (meal-wise, date-wise)
├── Mess_Menu.csv               # Hostel menu information
├── Raw_Materials.csv           # Ingredients/Stock data
├── merged_mess_data_*.csv      # Premerged datasets for analysis
│
├── Mess_app.py                 # Main Streamlit app with analytics, forecasting, and dashboard
├── attendance_model.pkl        # Trained attendance forecasting model
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation
│
├── .devcontainer/              # [dev container config] (if using Codespaces)
├── .venv/                      # [virtual environment]
```

---

## 📊 Key Components

### 1. Attendance Forecasting (Holt-Winters)
- Uses additive seasonal model for daily/weekly seasonality
- Captures trends & produces 7-day forecast with confidence intervals
- Achieves ~90% prediction accuracy on attendance

### 2. Waste Analysis
- Computes *waste per plate* (grams)
- Identifies high-waste meals/days
- Provides optimal portion recommendations to minimize waste

### 3. Two-Way ANOVA
- Evaluates factors: **Meal Type** and **Day**
- Results:
  - Meal Type: Significant (p < 0.05)
  - Day: Not significant
  - Interaction: Not significant

### 4. Streamlit Dashboard
- **Upload datasets** directly in-app for flexible usage
- Provides:
  - Attendance & waste trend visualizations
  - Menu lookup
  - Forecasts & recommendations
  - Correlation analysis heatmaps

---

## 🧪 How to Run the Project

### 1. Install Dependencies
```
pip install -r requirements.txt
```
If `requirements.txt` is missing, run:
```
pip install streamlit pandas numpy statsmodels matplotlib seaborn scikit-learn
```

### 2. Run the Streamlit App
```
streamlit run Mess_app.py
```

### 3. Upload Data
Upload these files in the dashboard:
- Mess_Attendance_Data.csv
- Food_Waste_data.csv
- Mess_Menu.csv

---

## 📈 Results

- ✔ 20% reduction in estimated food waste
- ✔ Improved meal planning accuracy via forecasting
- ✔ Decision support for supervisors (real-world Industrial Engineering)
- ✔ Intuitive, visual dashboard for usage by non-programmers

---

## ⚙️ Future Enhancements

- Add IoT attendance sensors (real-time data)
- Cost optimization with Operations Research
- Push notifications for mess updates
- Multi-hostel deployment (Cloud/Server)

---

## 🧑‍💻 Author

**Sevanth Kumar J**  
B.E. Industrial Engineering  
College of Engineering Guindy, Anna University

---

## ⭐ Support

If you found this project helpful, leave a ⭐ on GitHub!  
*Your support motivates me to keep building useful systems! 😊*

---
