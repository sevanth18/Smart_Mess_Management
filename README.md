📘 Smart Hostel Mess Management System
Forecasting Attendance & Reducing Food Waste Using Data Analytics

A data-driven system designed to help hostel messes predict student attendance, reduce food waste, and optimize portion planning using forecasting models, statistical analysis, and interactive visual dashboards.

🚀 Features
📊 1. Attendance Forecasting

Holt-Winters Exponential Smoothing

Captures trend + weekly seasonality

~90% prediction accuracy

🧪 2. Two-Way ANOVA

Analyzes impact of Day and Meal Type

Identifies statistically significant factors

🍽️ 3. Portion Optimization

Computes waste per plate

Suggests optimal portion size reductions

📈 4. Interactive Visualizations

Attendance trends

Waste patterns

Correlation heatmap

🌐 5. Streamlit Web App

Real-time forecasting

Dynamic analytics dashboard

♻️ 6. Food Waste Reduction

Achieved 15–20% reduction in estimated waste

🛠️ Tech Stack
Programming Language

Python

Libraries Used

Pandas, NumPy — data handling

Statsmodels — Holt-Winters forecasting & ANOVA

Matplotlib, Seaborn — charts

Streamlit — interactive dashboard

Scikit-learn — preprocessing & ML pipeline


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

Captures daily seasonal patterns

Forecasts next 7 days with CI

~90% accurate predictions

2. Waste Analysis

Computes waste per plate (grams)

Highlights high-waste meals

Supports portion size recommendations

3. Two-Way ANOVA Results
Factors Analyzed

Meal Type

Day of the Week

Outcome

✔ Meal Type → Significant (p < 0.05)

✘ Day → Not significant

✘ Interaction (Day × Meal) → Not significant

4. Streamlit Dashboard Features

Attendance trend visualization

Waste per plate graph

Menu lookup by Day × Meal

Forecasted attendance for next 7 days

Portion recommendation engine

🧪 How to Run the Project
1. Install Dependencies
pip install -r requirements.txt


If you do not have a requirements.txt:

pip install streamlit pandas numpy statsmodels matplotlib seaborn scikit-learn

2. Run Streamlit App
streamlit run streamlit_mess_app_final.py

3. Upload Required Datasets

Attendance_Data.csv

Food_Wastage.csv

Mess_Menu.csv

📈 Results & Insights
✔ ~20% reduction in estimated food waste
✔ Improved meal planning accuracy
✔ Identified significant factors affecting attendance
✔ Built a real-time decision support dashboard
✔ Applied core Industrial Engineering principles:

Forecasting

Optimization

Statistical analysis

Lean waste reduction

⚙️ Future Enhancements

🔗 IoT-based real-time attendance tracking

📉 Cost optimization using OR models

🔔 Automated planning alerts

☁️ Cloud deployment for multi-hostel scalability

👨‍💻 Author

Sevanth Kumar J
B.E. Industrial Engineering
College of Engineering Guindy, Anna University

⭐ Support

If this project helped you, please consider leaving a ⭐ on GitHub!
Your support motivates me to build more such systems 😊
