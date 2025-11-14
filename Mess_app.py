# streamlit_mess_app_final.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="AI Based Hostel Mess Food Waste Reduction", layout="wide")
st.title("🍽️ Smart Mess Management System")

# ---------------------------
# Helper functions
# ---------------------------
def read_any(file_obj):
    """Read uploaded file object (csv or xlsx) into DataFrame"""
    if file_obj is None:
        return None
    name = file_obj.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file_obj)
    else:
        return pd.read_excel(file_obj)

def standardize_meal_colnames(df):
    """If df has Breakfast/Lunch/Dinner columns or Morning/Afternoon/Night, keep as Breakfast/Lunch/Dinner"""
    cols = df.columns.tolist()
    # unify known patterns
    rename_map = {}
    for c in cols:
        lc = c.lower()
        if 'morning' in lc:
            rename_map[c] = 'Breakfast'
        if 'afternoon' in lc:
            rename_map[c] = 'Lunch'
        if 'night' in lc or 'dinner' in lc:
            rename_map[c] = 'Dinner'
        if c.strip().lower() == 'breakfast':
            rename_map[c] = 'Breakfast'
        if c.strip().lower() == 'lunch':
            rename_map[c] = 'Lunch'
        if c.strip().lower() == 'dinner':
            rename_map[c] = 'Dinner'
        if c.strip().lower() == 'date':
            rename_map[c] = 'Date'
        if c.strip().lower() == 'day':
            rename_map[c] = 'Day'
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

def melt_att_waste(att_df, waste_df):
    """Convert wide Attendance/Waste tables into long format and merge"""
    # detect date & day
    for df in [att_df, waste_df]:
        if df is None:
            continue
        if 'Date' not in df.columns:
            # try common alternatives
            possible = [c for c in df.columns if 'date' in c.lower() or 'day' in c.lower()]
            if possible:
                df.rename(columns={possible[0]: 'Date'}, inplace=True)
    # Ensure Date col exists
    if att_df is None or waste_df is None:
        return pd.DataFrame()
    # standardize meal column names
    att_df = standardize_meal_colnames(att_df)
    waste_df = standardize_meal_colnames(waste_df)

    # The expected meal columns
    meal_cols = ['Breakfast', 'Lunch', 'Dinner']
    # Check presence; if missing, try to detect any columns likely meals
    def detect_meal_cols(df):
        found = [c for c in meal_cols if c in df.columns]
        if found:
            return found
        # fallback: any numeric columns except Date/Day
        nums = [c for c in df.columns if c not in ['Date','Day'] and pd.api.types.is_numeric_dtype(df[c])]
        return nums

    att_meals = detect_meal_cols(att_df)
    waste_meals = detect_meal_cols(waste_df)

    # Melt
    try:
        att_long = att_df.melt(id_vars=[c for c in att_df.columns if c in ['Date','Day']],
                               value_vars=att_meals, var_name='Meal', value_name='Attendance')
    except Exception:
        att_long = pd.DataFrame(columns=['Date','Day','Meal','Attendance'])
    try:
        waste_long = waste_df.melt(id_vars=[c for c in waste_df.columns if c in ['Date','Day']],
                                   value_vars=waste_meals, var_name='Meal', value_name='Waste_kg')
    except Exception:
        waste_long = pd.DataFrame(columns=['Date','Day','Meal','Waste_kg'])

    # Normalize
    att_long.rename(columns={'Date':'Date'}, inplace=True)
    waste_long.rename(columns={'Date':'Date'}, inplace=True)
    # Clean Meal strings
    att_long['Meal'] = att_long['Meal'].astype(str).str.strip().str.title()
    waste_long['Meal'] = waste_long['Meal'].astype(str).str.strip().str.title()

    # Convert types
    att_long['Date'] = pd.to_datetime(att_long['Date'], errors='coerce')
    waste_long['Date'] = pd.to_datetime(waste_long['Date'], errors='coerce')
    att_long['Attendance'] = pd.to_numeric(att_long['Attendance'], errors='coerce')
    waste_long['Waste_kg'] = pd.to_numeric(waste_long['Waste_kg'], errors='coerce')

    # Merge on Date + Meal; Day can be taken from attendance if present
    df = pd.merge(att_long, waste_long, on=['Date','Meal'], how='left')
    # If Day missing, create from Date
    if 'Day' not in df.columns or df['Day'].isna().all():
        df['Day'] = df['Date'].dt.day_name()
    else:
        # Normalize Day column
        df['Day'] = df['Day'].fillna(df['Date'].dt.day_name())
    # Compute waste per plate grams
    df['Waste_per_plate_g'] = (df['Waste_kg'] / df['Attendance']) * 1000
    # If Attendance zero or NaN, set waste_per_plate to NaN
    df.loc[(df['Attendance'] <= 0) | (df['Attendance'].isna()), 'Waste_per_plate_g'] = np.nan
    # Sort
    df = df.sort_values('Date').reset_index(drop=True)
    return df

def ensure_minimum_rows(df, min_rows=14):
    """Return True if df has sufficient rows for time series forecasting"""
    return len(df.dropna(subset=['Attendance'])) >= min_rows

def holt_winters_forecast(series, forecast_periods=7):
    """Fit Holt-Winters and forecast; return forecast series and ci computed from residuals"""
    # series: pd.Series indexed by date
    if len(series.dropna()) < 8:
        raise ValueError("Not enough data for Holt-Winters (need >=8 non-null rows).")
    # choose additive seasonal with weekly period
    try:
        model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=7, initialization_method="estimated")
        fit = model.fit(optimized=True)
        forecast = fit.forecast(forecast_periods)
        # simple CI using residual std
        resid = fit.resid.dropna()
        se = resid.std() if len(resid) > 1 else resid.values.std() if len(resid)>0 else 0
        lower = forecast - 1.96 * se
        upper = forecast + 1.96 * se
        return fit.fittedvalues, forecast, lower, upper
    except Exception as e:
        raise e

# ---------------------------
# Sidebar: File upload or local fallback
# ---------------------------
st.sidebar.header("Upload datasets (or leave blank to use local files if present)")

attendance_file = st.sidebar.file_uploader("Attendance_Data (CSV or XLSX)", type=["csv","xlsx"])
waste_file = st.sidebar.file_uploader("Food_Wastage (CSV or XLSX)", type=["csv","xlsx"])
menu_file = st.sidebar.file_uploader("Mess_Menu (CSV or XLSX)", type=["csv","xlsx"])

# Fallback to local file names if not uploaded
if attendance_file is None:
    try:
        df_att_upload = pd.read_csv("Attendance_Data.csv")
        st.sidebar.info("Loaded local Attendance_Data.csv")
    except Exception:
        df_att_upload = None
else:
    df_att_upload = read_any(attendance_file)

if waste_file is None:
    try:
        df_waste_upload = pd.read_csv("Food_Wastage.csv")
        st.sidebar.info("Loaded local Food_Wastage.csv")
    except Exception:
        df_waste_upload = None
else:
    df_waste_upload = read_any(waste_file)

if menu_file is None:
    try:
        df_menu_upload = pd.read_csv("Mess_Menu.csv")
        st.sidebar.info("Loaded local Mess_Menu.csv")
    except Exception:
        df_menu_upload = None
else:
    df_menu_upload = read_any(menu_file)

# Validate basic presence
if df_att_upload is None or df_waste_upload is None:
    st.warning("Please upload Attendance_Data and Food_Wastage files (or place Attendance_Data.csv and Food_Wastage.csv in the app folder).")
    st.stop()

# ---------------------------
# Data preparation
# ---------------------------
# Standardize colnames to expected
df_att = standardize_meal_colnames(df_att_upload.copy())
df_waste = standardize_meal_colnames(df_waste_upload.copy())
df_menu = df_menu_upload.copy() if df_menu_upload is not None else pd.DataFrame(columns=['Day','Meal','Menu_Item'])

# Melt and merge
df_long = melt_att_waste(df_att, df_waste)

# Merge menu: our menu has Day, Meal, Menu_Item (user said Mess_Menu.csv has Day,Meal,Menu_Item)
if not df_menu.empty:
    # Normalize menu
    df_menu = df_menu.rename(columns={c:c.strip().title() for c in df_menu.columns})
    # ensure columns names Day, Meal, Menu_Item
    if 'Day' not in df_menu.columns and 'Date' in df_menu.columns:
        df_menu['Day'] = pd.to_datetime(df_menu['Date'], errors='coerce').dt.day_name()
    # Standardize Meal wording
    if 'Meal' in df_menu.columns:
        df_menu['Meal'] = df_menu['Meal'].astype(str).str.strip().str.title()
    # Merge by Day & Meal
    df_long = pd.merge(df_long, df_menu[['Day','Meal','Menu_Item']].drop_duplicates(), on=['Day','Meal'], how='left')

# Derived features
df_long['dow'] = df_long['Date'].dt.day_name()
df_long['Date'] = pd.to_datetime(df_long['Date'])
# fill small missing Day values
df_long['Day'] = df_long['Day'].fillna(df_long['dow'])

# Sidebar quick stats
st.sidebar.subheader("Dataset Summary")
st.sidebar.write("Start date:", str(df_long['Date'].min().date() if not df_long['Date'].isna().all() else "N/A"))
st.sidebar.write("End date:", str(df_long['Date'].max().date() if not df_long['Date'].isna().all() else "N/A"))
st.sidebar.write("Records (rows):", len(df_long))
st.sidebar.write("Meals detected:", ', '.join(sorted(df_long['Meal'].dropna().unique())))

# ---------------------------
# Main layout - top KPIs
# ---------------------------
st.subheader("Overview / KPIs")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Days Recorded", df_long['Date'].nunique())
col2.metric("Average Attendance (all)", round(df_long['Attendance'].mean(skipna=True) or 0,2))
col3.metric("Average Waste per plate (g)", round(df_long['Waste_per_plate_g'].mean(skipna=True) or 0,2))
col4.metric("Menu entries (unique)", int(df_long['Menu_Item'].nunique() if 'Menu_Item' in df_long.columns else 0))

# Preview merged dataset
with st.expander("Preview merged dataset (Attendance + Waste + Menu)"):
    st.dataframe(df_long.head(20))

# ---------------------------
# Attendance & Waste Trends
# ---------------------------
st.write("## Attendance & Waste Trends")
# Pivot attendance to wide: index Date, columns Meal
pivot_att = df_long.pivot_table(index='Date', columns='Meal', values='Attendance', aggfunc='mean')
pivot_waste = df_long.pivot_table(index='Date', columns='Meal', values='Waste_per_plate_g', aggfunc='mean')

# Attendance chart
st.write("### Attendance trends (per meal)")
if pivot_att.empty or pivot_att.dropna(how='all').empty:
    st.warning("No attendance data available to plot. Check your Attendance_Data.csv")
else:
    st.line_chart(pivot_att.fillna(method='ffill'))

# Waste chart
st.write("### Waste per plate (g) trends")
if pivot_waste.empty or pivot_waste.dropna(how='all').empty:
    st.warning("No waste data available to plot. Check your Food_Wastage.csv")
else:
    st.line_chart(pivot_waste.fillna(method='ffill'))

# Average attendance by day
st.write("### Average Attendance by Day of Week")
avg_by_day = df_long.groupby('Day').agg(avg_att=('Attendance','mean')).reindex(
    ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']).reset_index()
fig, ax = plt.subplots(figsize=(8,3))
sns.barplot(data=avg_by_day, x='Day', y='avg_att', ax=ax)
ax.set_ylabel("Avg Attendance")
plt.xticks(rotation=45)
st.pyplot(fig)

# Correlation heatmap (attendance across meals)
st.write("### Correlation between meals (attendance)")
if pivot_att.shape[1] >= 2:
    corr = pivot_att.corr()
    fig, ax = plt.subplots(figsize=(4,3))
    sns.heatmap(corr, annot=True, vmin=-1, vmax=1, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    # show interpretation
    st.caption("Higher correlation suggests attendance at one meal relates to another. Use this for deeper analysis.")
else:
    st.info("Need at least two meal series to compute correlation heatmap.")

# ---------------------------
# Forecasting (Holt-Winters) per meal
# ---------------------------
st.write("## Forecasting Attendance (Holt-Winters)")

forecast_meal = st.selectbox("Choose meal to forecast", options=sorted(df_long['Meal'].dropna().unique()))
forecast_horizon = st.number_input("Forecast horizon (days)", min_value=1, max_value=30, value=7)

# build time series for selected meal
ts = df_long[df_long['Meal']==forecast_meal].set_index('Date').sort_index()['Attendance'].asfreq('D')
# fill small gaps by forward fill (not changing historical numbers)
ts = ts.fillna(method='ffill')

if ts.dropna().empty or len(ts.dropna()) < 8:
    st.warning("Not enough historical daily attendance data for this meal to run Holt-Winters. Need >=8 days.")
else:
    try:
        fitted_vals, forecast_vals, lower_ci, upper_ci = holt_winters_forecast(ts.dropna(), forecast_periods=int(forecast_horizon))
        # Plot: actual, fitted, forecast with CI
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(ts.index, ts.values, label='Actual', marker='.', linewidth=1)
        ax.plot(fitted_vals.index, fitted_vals.values, label='Fitted', linestyle='--')
        # forecast index
        last_date = ts.dropna().index.max()
        f_index = pd.date_range(last_date + pd.Timedelta(days=1), periods=len(forecast_vals), freq='D')
        ax.plot(f_index, forecast_vals.values, label='Forecast', marker='o')
        ax.fill_between(f_index, lower_ci.values, upper_ci.values, color='gray', alpha=0.25, label='Approx. 95% CI')
        ax.set_title(f"{forecast_meal} attendance forecast")
        ax.set_ylabel("Attendance")
        ax.legend()
        st.pyplot(fig)

        # show numeric forecast
        df_forecast = pd.DataFrame({
            'Date': f_index,
            'Forecast_Attendance': np.round(forecast_vals.values).astype(int),
            'Lower_CI': np.round(lower_ci.values).astype(int),
            'Upper_CI': np.round(upper_ci.values).astype(int)
        })
        st.dataframe(df_forecast.reset_index(drop=True))
    except Exception as e:
        st.error(f"Forecast failed: {e}")

# ---------------------------
# Two-way ANOVA (Day x Meal)
# ---------------------------
st.write("## Two-way ANOVA: Does Day & Meal affect Attendance?")
with st.expander("Run Two-way ANOVA and view results"):
    anova_df = df_long.dropna(subset=['Attendance']).copy()
    # ensure categorical
    anova_df['Day'] = anova_df['Day'].astype(str)
    anova_df['Meal'] = anova_df['Meal'].astype(str)
    if anova_df.empty or anova_df['Attendance'].isna().all():
        st.warning("Not enough attendance data to run ANOVA.")
    else:
        try:
            model = ols('Attendance ~ C(Day) + C(Meal) + C(Day):C(Meal)', data=anova_df).fit()
            aov_table = sm.stats.anova_lm(model, typ=2)  # two-way
            st.write("ANOVA table (Type II):")
            st.dataframe(aov_table.style.format({"sum_sq":"{:.2f}", "F":"{:.2f}", "PR(>F)":"{:.4f}"}))
            st.write("Interpretation:")
            st.write("- If p-value (PR(>F)) < 0.05 for a factor, the factor has a statistically significant effect on attendance.")
        except Exception as e:
            st.error(f"ANOVA failed: {e}")

# ---------------------------
# Menu recommendation + historical menu analysis
# ---------------------------
st.write("## Menu Lookup & Historical Performance")
colA, colB = st.columns([2,1])
with colA:
    sel_date = st.date_input("Select date to view menu", value=datetime.today())
    sel_meal = st.selectbox("Select meal", options=sorted(df_long['Meal'].dropna().unique()))
    # Find menu item
    menu_item = None
    if 'Menu_Item' in df_long.columns:
        row = df_long[(df_long['Day'] == pd.to_datetime(sel_date).day_name()) & (df_long['Meal']==sel_meal)]
        if not row.empty and row['Menu_Item'].notna().any():
            menu_item = row['Menu_Item'].iloc[0]
    if menu_item:
        st.success(f"Menu for {sel_meal} on {sel_date}: **{menu_item}**")
    else:
        st.info("No menu item found for this selection in Mess_Menu.csv (or Mess_Menu not provided).")

with colB:
    st.write("Historical attendance for selected menu / meal")
    # If menu exists, show historical attendance for same menu item
    if 'Menu_Item' in df_long.columns and menu_item:
        hist = df_long[df_long['Menu_Item']==menu_item].sort_values('Date', ascending=False)[['Date','Meal','Attendance','Waste_per_plate_g']].head(10)
        st.dataframe(hist)
    else:
        # show recent attendance for same meal
        hist2 = df_long[df_long['Meal']==sel_meal].sort_values('Date', ascending=False)[['Date','Meal','Attendance','Waste_per_plate_g']].head(10)
        st.dataframe(hist2)

# ---------------------------
# Portion recommendation & ingredient scaling (basic)
# ---------------------------
st.write("## Portion Recommendation & Cooking Quantity")
pr_col1, pr_col2 = st.columns(2)
with pr_col1:
    selected_meal_for_plan = st.selectbox("Choose meal for planning", options=sorted(df_long['Meal'].dropna().unique()), index=0)
    selected_date_for_plan = st.date_input("Date for plan", value=(datetime.today() + timedelta(days=1)))
    default_portion_g = st.number_input("Default portion per plate (g)", min_value=50, max_value=1000, value=250, step=10)
    acceptable_waste_g = st.slider("Acceptable waste per plate (g)", min_value=0, max_value=300, value=30)
with pr_col2:
    # Predict attendance (use Holt-Winters forecast if available else median)
    # Try to use existing forecast if same meal was forecasted above
    # Build simple median fallback
    recent_series = df_long[df_long['Meal']==selected_meal_for_plan].set_index('Date').sort_index()['Attendance']
    predicted_att = int(round(recent_series.dropna().tail(7).mean())) if not recent_series.dropna().empty else 0
    # If model forecast done earlier and meal matches, use last forecast row if present
    try:
        if forecast_meal == selected_meal_for_plan and 'df_forecast' in locals():
            # take first forecast value
            predicted_att = int(df_forecast['Forecast_Attendance'].iloc[0])
    except Exception:
        pass
    st.write("Predicted attendance (simple):", int(predicted_att))
    suggested_total_kg = round((default_portion_g / 1000.0) * predicted_att, 2)
    st.metric("Suggested total to cook (kg)", suggested_total_kg)

# Portion adjustment using average waste per plate
recent_meal = df_long[df_long['Meal']==selected_meal_for_plan].sort_values('Date', ascending=False).head(30)
avg_waste_g = recent_meal['Waste_per_plate_g'].mean()
if np.isnan(avg_waste_g):
    st.info("Not enough waste data for this meal to compute adaptive portioning.")
else:
    st.write(f"Recent avg waste per plate for {selected_meal_for_plan}: {avg_waste_g:.1f} g")
    if avg_waste_g > acceptable_waste_g:
        reduce_pct = min(0.25, (avg_waste_g - acceptable_waste_g) / (avg_waste_g + 1e-6))
        new_portion = int(default_portion_g * (1 - reduce_pct))
        st.warning(f"Avg waste exceeds acceptable. Suggest reducing portion: {default_portion_g}g → {new_portion}g ({reduce_pct*100:.1f}% reduction).")
    else:
        st.success("Average waste is within acceptable target. Keep current portion.")

# If raw materials were uploaded earlier (optional), scale ingredients
if 'Raw_Materials.csv' in st.sidebar.text_input("Optional: type 'include raw' to enable raw-material scaling (not required)", value="").lower():
    st.info("Raw material scaling requested but feature is optional in this version. Provide Raw_Materials.csv via upload to enable automatic scaling.")

# ---------------------------
# Export options
# ---------------------------
st.write("## Export / Save")
if st.button("Save cleaned merged dataset to CSV (local)"):
    out_name = f"merged_mess_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df_long.to_csv(out_name, index=False)
    st.success(f"Saved merged dataset as {out_name} in current folder.")

st.caption("App created to demonstrate forecasting, ANOVA, menu lookup, waste & portion analysis. Replace uploaded CSVs with your real data for accurate insights.")
