import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.metrics import mean_squared_error

# Load datasets
k11 = pd.read_csv("data/Kharif/Kharif11-12.csv")
k12 = pd.read_csv("data/Kharif/Kharif12-13.csv")
k13 = pd.read_csv("data/Kharif/Kharif13-14.csv")
k14 = pd.read_csv("data/Kharif/Kharif14-15.csv")
k15 = pd.read_csv("data/Kharif/Kharif15-16.csv")
k16 = pd.read_csv("data/Kharif/Kharif16-17.csv")
k17 = pd.read_csv("data/Kharif/Kharif17-18.csv")
k18 = pd.read_csv("data/Kharif/Kharif18-19.csv")
k19 = pd.read_csv("data/Kharif/Kharif19-20.csv")
k20 = pd.read_csv("data/Kharif/Kharif20-21.csv")
k21 = pd.read_csv("data/Kharif/kharif21-22.csv")
r11 = pd.read_csv("data/Rabi/Rabi11-12.csv")
r12 = pd.read_csv("data/Rabi/Rabi12-13.csv")
r13 = pd.read_csv("data/Rabi/Rabi13-14.csv")
r14 = pd.read_csv("data/Rabi/Rabi14-15.csv")
r15 = pd.read_csv("data/Rabi/Rabi15-16.csv")
r16 = pd.read_csv("data/Rabi/Rabi16-17.csv")
r17 = pd.read_csv("data/Rabi/Rabi17-18.csv")
r18 = pd.read_csv("data/Rabi/Rabi18-19.csv")
r19 = pd.read_csv("data/Rabi/Rabi19-20.csv")
r20 = pd.read_csv("data/Rabi/Rabi20-21.csv")
r21 = pd.read_csv("data/Rabi/Rabi21-22.csv")

# Kharif
static_value = 'Kharif'
frames = [k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21]
df = pd.concat(frames, ignore_index=True)
df['Season'] = static_value
sorted_df = df.sort_values(by='DISTRICT', ascending=True).reset_index(drop=True)

# Rabi
static_value2 = 'Rabi'
frames2 = [r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21]
df1 = pd.concat(frames2, ignore_index=True)
df1['Season'] = static_value2
sorted_df1 = df1.sort_values(by='DISTRICT', ascending=True).reset_index(drop=True)

# Combine datasets
frames3 = [sorted_df, sorted_df1]
df_final = pd.concat(frames3, ignore_index=True)

# Data preprocessing
data_encoded = pd.get_dummies(df_final, columns=['Season'], drop_first=True)
data = pd.get_dummies(data_encoded, columns=['DISTRICT'])

X_n = data.drop(['NITROGENIUS', 'TOTAL'], axis=1)
Y_n = data['NITROGENIUS']
X_p = data.drop(['PHOSPHETIC', 'TOTAL'], axis=1)
Y_p = data['PHOSPHETIC']
X_k = data.drop(['POTASIC', 'TOTAL'], axis=1)
Y_k = data['POTASIC']

# Load models
model_n = joblib.load('model_n.pkl')
model_p = joblib.load('model_p.pkl')
model_k = joblib.load('model_k.pkl')

# Streamlit app
st.title("Fertilizer Recommendation System")

st.markdown(
    """
    <style>
        body {
            background-image: url("fertilizer.jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            opacity: 0.8;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Input fields
district = st.selectbox("Select District", df_final['DISTRICT'].unique())
season = st.selectbox("Select Season", ['Kharif', 'Rabi'])
nitrogen = st.number_input("Enter current Nitrogen level", min_value=0.0, max_value=10000.0, step=1.0)
phosphorus = st.number_input("Enter current Phosphorus level", min_value=0.0, max_value=10000.0, step=1.0)
potash = st.number_input("Enter current Potash level", min_value=0.0, max_value=10000.0, step=1.0)

# Prepare input data
input_data = {
    'NITROGENIUS': nitrogen,
    'PHOSPHETIC': phosphorus,
    'POTASIC': potash,
    'Season_Kharif': 1 if season == 'Kharif' else 0
}
for dist in df_final['DISTRICT'].unique():
    input_data[f'DISTRICT_{dist}'] = 1 if dist == district else 0

input_df = pd.DataFrame([input_data])

# Align input data with training data
input_df = input_df.reindex(columns=X_n.columns, fill_value=0)

# Predict
if st.button("Predict"):
    pred_n = model_n.predict(input_df)
    pred_p = model_p.predict(input_df)
    pred_k = model_k.predict(input_df)

    st.success(f"Predicted Nitrogen Level: {pred_n[0]:.2f}")
    st.success(f"Predicted Phosphorus Level: {pred_p[0]:.2f}")
    st.success(f"Predicted Potash Level: {pred_k[0]:.2f}")

    delta_n = pred_n[0] - nitrogen
    delta_p = pred_p[0] - phosphorus
    delta_k = pred_k[0] - potash

    if delta_n > 0:
        st.info(f"Amount of Nitrogen to add: {delta_n:.2f}")
    else:
        st.info(f"Amount of Nitrogen to reduce: {abs(delta_n):.2f}")
    
    if delta_p > 0:
        st.info(f"Amount of Phosphorus to add: {delta_p:.2f}")
    else:
        st.info(f"Amount of Phosphorus to reduce: {abs(delta_p):.2f}")
    
    if delta_k > 0:
        st.info(f"Amount of Potash to add: {delta_k:.2f}")
    else:
        st.info(f"Amount of Potash to reduce: {abs(delta_k):.2f}")
