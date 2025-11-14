📘 Smart Hostel Mess Management System
Forecasting Attendance & Optimizing Food Portions using Data Analytics

A data-driven system designed to help hostel messes predict student attendance, reduce food waste, and optimize portion planning using forecasting, statistical analysis, and interactive dashboards.

🚀 Features

📊 Meal-wise Attendance Forecasting using Holt-Winters Exponential Smoothing

🧪 Two-Way ANOVA to analyze the impact of Day and Meal Type

🍽️ Portion Optimization based on waste per plate

📈 Interactive Visualizations for attendance, waste trends, and correlations

🌐 Streamlit Web App for real-time decision support

♻️ 15–20% Food Waste Reduction using analytics-driven planning

🛠️ Tech Stack

Programming Language: Python
Libraries Used:

Pandas, NumPy → Data handling

Statsmodels → Forecasting & ANOVA

Matplotlib, Seaborn → Visualizations

Streamlit → Dashboard UI

Scikit-learn → Preprocessing and ML pipeline

📂 Project Structure
mini_project/
│
├── Attendance_Data.csv
├── Food_Wastage.csv
├── Mess_Menu.csv
│
├── streamlit_mess_app_final.py
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

Generates next 7-day forecast with confidence intervals

2. Waste Analysis

Computes waste per plate (grams)

Identifies high-waste meals

Suggests optimal portion reduction

3. Two-Way ANOVA

Factors studied: Meal Type and Day

Results:

Meal Type → Significant impact (p < 0.05)

Day → Not significant

Interaction → Not significant

4. Streamlit Dashboard

Provides:

Attendance & waste trends

Menu lookup

Forecasting visualizations

Portion recommendations

🧪 How to Run the Project
1. Install Dependencies
pip install -r requirements.txt


If you don't have a requirements.txt, use:

pip install streamlit pandas numpy statsmodels matplotlib seaborn scikit-learn

2. Run Streamlit App
streamlit run streamlit_mess_app_final.py

3. Upload Datasets

Inside the dashboard, upload:

Attendance_Data.csv

Food_Wastage.csv

Mess_Menu.csv

📈 Results

✔ Achieved 20% reduction in estimated food waste

✔ Improved meal planning accuracy using forecasting

✔ Provided a decision-support tool for mess supervisors

✔ Demonstrated real-world application of Industrial Engineering principles

⚙️ Future Enhancements

Add IoT sensors for real-time attendance tracking

Cost optimization using OR techniques

Push notifications for mess planning updates

Cloud deployment for multi-hostel scalability

🧑‍💻 Author

Sevanth Kumar J
B.E. Industrial Engineering
College of Engineering Guindy, Anna University

⭐ Support

If you like this project, please leave a ⭐ on GitHub!
It motivates me to build more such systems 😊
