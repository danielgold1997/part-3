import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
import pickle
import car_data_prep

# Load the data
df = pd.read_csv('dataset.csv')

# Prepare the data
df_prepared = car_data_prep.prepare_data(df)

# Separate features and target
X = df_prepared.drop('Price', axis=1)
y = df_prepared['Price']

# Save the column names
model_columns = X.columns.tolist()

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the model
model = ElasticNet()

# Train the model
model.fit(X_train, y_train)

# Save the trained model to a file
with open('trained_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Save the column names
with open('model_columns.pkl', 'wb') as file:
    pickle.dump(model_columns, file)

# Save the scaler
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
