import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# 1. Load Data
print("Loading train.csv...")
train_df = pd.read_csv('train.csv')

# 2. Feature Engineering
print("Engineering features...")
train_df['Temp_Diff'] = train_df['Process temperature [K]'] - train_df['Air temperature [K]']
train_df['Power_Load'] = train_df['Rotational speed [rpm]'] * train_df['Torque [Nm]']
train_df['Type'] = train_df['Type'].map({'L': 0, 'M': 1, 'H': 2})

# Drop non-predictive & failure sub-type columns
drop_cols = ['id', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
X = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
y = train_df['Machine failure']

# 3. Scale Features
print("Scaling features...")
scale_cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 
              'Torque [Nm]', 'Tool wear [min]', 'Temp_Diff', 'Power_Load']
scaler = StandardScaler()
X[scale_cols] = scaler.fit_transform(X[scale_cols])

# 4. Train Model
print("Training Random Forest model (this might take a few seconds)...")
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X, y)

# 5. Save Model and Scaler
joblib.dump(model, 'rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("✅ SUCCESS: rf_model.pkl and scaler.pkl have been created!")