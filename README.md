📘 Smart Hostel Mess Management System
Forecasting Attendance & Reducing Food Waste using Data Analytics

A data-driven system designed to help hostel messes predict student attendance, reduce food waste, and optimize portion planning using forecasting, statistical analysis, and interactive dashboards.

🚀 Features
📊 Meal-wise Attendance Forecasting using Holt-Winters Exponential Smoothing
🧪 Two-Way ANOVA to analyze the impact of Day and Meal Type
🍽️ Portion Optimization based on waste per plate
📈 Interactive Visualizations for attendance, waste trends, and correlations
🌐 Streamlit Web App for real-time decision support
♻️ 15–20% Reduction in Food Waste using analytics-driven planning

🛠️ Tech Stack
1.Programming Language:
2.Python
3.Libraries Used:
                 -->Pandas, NumPy → Data handling
                 -->Statsmodels → Forecasting & ANOVA
                 -->Matplotlib, Seaborn → Visualizations
                 -->Streamlit → Dashboard interface
                 --Scikit-learn → Preprocessing & ML pipeline

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
Uses additive seasonal model
Captures trend + weekly seasonality
Achieved ≈90% prediction accuracy
Forecasts next 7 days with confidence intervals

2. Waste Analysis
Computes waste per plate (in grams)
Identifies high-waste meals
Suggests optimal portion reduction

3. Two-Way ANOVA
Factors analyzed:
Meal Type
Day of the Week

Results:
✔ Meal Type → Significant (p < 0.05)
✘ Day → Not significant
✘ Interaction (Day × Meal) → Not significant

4. Streamlit Dashboard
Provides the following insights:
Attendance & waste trends
Menu lookup by day & meal
Forecasted attendance
Portion recommendations


🧪 How to Run the Project
1. Install Dependencies
pip install -r requirements.txt

If you don’t have a requirements file:
pip install streamlit pandas numpy statsmodels matplotlib seaborn scikit-learn

2. Run the Streamlit Application
streamlit run streamlit_mess_app_final.py

3. Upload the Datasets in the App
Attendance_Data.csv
Food_Wastage.csv
Mess_Menu.csv



📈 Results
✔ Achieved ~20% reduction in estimated food waste
✔ Improved meal planning accuracy using forecasting
✔ Built a real-time decision support tool for mess supervisors
✔ Demonstrated practical application of Industrial Engineering principles

1.Forecasting
2.Optimization
3.Statistical analysis
4.Process improvement

⚙️ Future Enhancements
IoT-based real-time attendance tracking
Cost optimization using OR techniques
Automated alerts and notifications
Cloud deployment for multi-hostel usage

👨‍💻 Author
Sevanth Kumar J
B.E. Industrial Engineering
College of Engineering Guindy, Anna University

⭐ Support
If you found this project helpful, please consider giving it a ⭐ on GitHub.
Your support motivates me to build more such systems! 😊
