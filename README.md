📘 Smart Hostel Mess Management System
Forecasting Attendance & Reducing Food Waste Using Data Analytics

A data-driven system designed to help hostel messes predict student attendance, reduce food waste, and optimize portion planning using forecasting models, statistical analysis, and interactive dashboards.

🚀 Features
📊 1. Attendance Forecasting

Uses Holt-Winters Exponential Smoothing

Captures trend + weekly seasonality

Achieves ~90% prediction accuracy

🧪 2. Two-Way ANOVA

Analyzes the impact of Day and Meal Type

Meal Type is statistically significant (p < 0.05)

Helps understand attendance variability

🍽️ 3. Portion Optimization

Computes waste per plate (grams)

Suggests optimized portion sizes

Enables 15–20% reduction in food waste

📈 4. Interactive Visualizations

Attendance trends

Waste per plate patterns

Correlation heatmaps

🌐 5. Streamlit Dashboard

Real-time forecasting

Dynamic UI for ease of use

Designed for mess supervisors

🛠️ Tech Stack
Programming Language

Python

Libraries Used

Pandas, NumPy → Data handling

Statsmodels → Forecasting & ANOVA

Matplotlib, Seaborn → Visualizations

Streamlit → Dashboard UI

Scikit-learn → Preprocessing & ML utilities

📂 Project Structure
mini_project/
│
├── Attendance_Data.csv
├── Food_Wastage.csv
├── Mess_Menu.csv
│
├── streamlit_mess_app_final.py
│
├── charts/
│   ├── attendance_trend.png
│   ├── waste_trend.png
│   ├── forecast.png
│   └── correlation_heatmap.png
│
└── README.md

📊 Key Components
1. Attendance Forecasting (Holt-Winters)

Additive seasonal model

Captures meal-wise daily patterns

Predicts next 7 days

Provides confidence intervals

2. Waste Analysis

Computes waste per plate

Identifies high-waste meals

Supports portion-based decision making

3. Two-Way ANOVA

Factors analyzed:

Meal Type

Day of the Week

Results:

Meal Type → Significant (p < 0.05)

Day → Not significant

Interaction → Not significant

4. Streamlit Dashboard Features

Attendance & waste trend charts

Menu lookup by Day × Meal

Forecasting graph with CI

Portion recommendation engine

🧪 How to Run the Project
1. Install Dependencies
pip install -r requirements.txt


If you don’t have a requirements file:

pip install streamlit pandas numpy statsmodels matplotlib seaborn scikit-learn

2. Run the Streamlit App
streamlit run streamlit_mess_app_final.py

3. Upload the Required Datasets

Attendance_Data.csv

Food_Wastage.csv

Mess_Menu.csv

📈 Results

Achieved 15–20% reduction in predicted food waste

Improved attendance planning accuracy

Identified key factors affecting daily and meal-wise attendance

Delivered a real-time decision support dashboard

Applied Industrial Engineering concepts:

Forecasting

Optimization

Statistical analysis

Lean waste reduction

⚙️ Future Enhancements

IoT-based real-time attendance tracking

Cost optimization using OR models

Automated alerts & notifications

Multi-hostel cloud deployment

👨‍💻 Author

Sevanth Kumar J
B.E. Industrial Engineering
College of Engineering Guindy, Anna University

⭐ Support

If you found this project helpful, please consider giving it a ⭐ on GitHub.
Your support motivates me to build more such systems 😊
