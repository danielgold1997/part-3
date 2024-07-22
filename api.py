from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import pandas as pd
import car_data_prep

app = Flask(__name__)
CORS(app)

# Load the model
with open('trained_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the model columns
with open('model_columns.pkl', 'rb') as file:
    model_columns = pickle.load(file)

# Load the scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print("Data received:", data)
    
    # בדיקה אם הנתונים הם רשימה של מילונים, במידה ולא לעטוף אותם ברשימה
    if isinstance(data, dict):
        data = [data]
    
    # המרת הנתונים ל- DataFrame
    df = pd.DataFrame(data)
    print("DataFrame created:", df)
    
    try:
        # הפעלת פונקציית prepare_data
        df_prepared = car_data_prep.prepare_data(df)
        print("Prepared DataFrame:", df_prepared)
        
        # הבטחת תאימות העמודות עם המודל
        df_prepared = df_prepared.reindex(columns=model_columns, fill_value=0)
        print("Aligned DataFrame:", df_prepared)
        
        # Scale the data
        df_scaled = scaler.transform(df_prepared)
        
    except KeyError as e:
        return jsonify({'error': str(e)}), 400
    
    # קבלת ניבוי מהמודל
    try:
        prediction = model.predict(df_scaled)
        print("Prediction:", prediction)
    except Exception as e:
        print("Prediction error:", str(e))
        return jsonify({'error': str(e)}), 500
    
    return jsonify({'predicted_price': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
