from flask import Flask,render_template,request, jsonify
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('model_p.pkl','rb'))
app = Flask(__name__)

df = pd.read_csv('NIFTY 50_Data.csv')

# Rename columns to ds and y (required by Prophet)
df = df.rename(columns={'Date': 'ds', 'Close': 'y'})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    data = request.get_json()

    # Create a DataFrame with the future dates for prediction
    future_dates = pd.date_range(start=df.ds.max(), periods=data['periods'], freq='D')
    future = pd.DataFrame({'ds': future_dates})

    # Make predictions with the model
    forecast = model.predict(future)

    # Select the relevant columns from the forecast DataFrame
    forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    # Convert the forecast DataFrame to a JSON object
    forecast_json = forecast.to_json(orient='records')

    return jsonify(forecast_json)


if __name__ == '__main__':
    app.run(debug=True)