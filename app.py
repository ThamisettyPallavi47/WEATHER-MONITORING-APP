
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
import os
import requests
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import json
import openai
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DIR = os.path.join(BASE_DIR, 'public')
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')

# Create directories if they don't exist
if not os.path.exists(PUBLIC_DIR):
    os.makedirs(PUBLIC_DIR)
if not os.path.exists(TEMPLATES_DIR):
    os.makedirs(TEMPLATES_DIR)

app = Flask(__name__, static_folder=PUBLIC_DIR, static_url_path='/static', template_folder=TEMPLATES_DIR)
CORS(app)

# API Keys from environment variables

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY


BASE_WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
AIR_QUALITY_URL = "http://api.openweathermap.org/data/2.5/air_pollution"
UV_INDEX_URL = "http://api.openweathermap.org/data/2.5/uvi"
FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"

# File paths
weather_data_file = "weather_data.json"
PREDICTION_MODELS_FILE = "prediction_models.pkl"

# ML Models
class WeatherPredictor:
    def __init__(self):
        self.precipitation_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.temperature_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.humidity_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.windspeed_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
        
    def train_models(self, X, y_precip, y_temp, y_humidity, y_windspeed):
        X_train, X_test, y_precip_train, y_precip_test = train_test_split(X, y_precip, test_size=0.2, random_state=42)
        _, _, y_temp_train, y_temp_test = train_test_split(X, y_temp, test_size=0.2, random_state=42)
        _, _, y_humidity_train, y_humidity_test = train_test_split(X, y_humidity, test_size=0.2, random_state=42)
        _, _, y_windspeed_train, y_windspeed_test = train_test_split(X, y_windspeed, test_size=0.2, random_state=42)
        
        self.precipitation_model.fit(X_train, y_precip_train)
        self.temperature_model.fit(X_train, y_temp_train)
        self.humidity_model.fit(X_train, y_humidity_train)
        self.windspeed_model.fit(X_train, y_windspeed_train)
        self.is_trained = True
        
    def predict(self, features, model_type):
        if not self.is_trained:
            return None
        if model_type == 'precipitation':
            return self.precipitation_model.predict([features])[0]
        elif model_type == 'temperature':
            return self.temperature_model.predict([features])[0]
        elif model_type == 'humidity':
            return self.humidity_model.predict([features])[0]
        elif model_type == 'windspeed':
            return self.windspeed_model.predict([features])[0]

predictor = WeatherPredictor()

def load_data():
    if os.path.exists(weather_data_file):
        with open(weather_data_file, "r") as f:
            return json.load(f)
    return {}

def save_data(data):
    with open(weather_data_file, "w") as f:
        json.dump(data, f, indent=2)

def get_wind_direction(degrees):
    """Convert wind degrees to cardinal direction for speech"""
    if degrees is None:
        return "unknown direction"
    directions = ['North', 'North-East', 'East', 'South-East', 
                 'South', 'South-West', 'West', 'North-West']
    return directions[round(degrees / 45) % 8]

@app.route('/')
def serve_index():
    return send_from_directory(PUBLIC_DIR,'index.html')


@app.route('/advanced_prediction')
def advanced_prediction_page():
    city = request.args.get('city', 'London')
    return render_template('advanced_prediction.html', city=city)



@app.route('/map')
def map_page():
    return render_template('map.html')

@app.route('/visualization')
def visualization_page():
    city = request.args.get('city', 'Hyderabad')
    return render_template('visualization.html', city=city)

@app.route('/weather_prediction')
def weather_prediction_page():
    city = request.args.get('city', 'London')
    return render_template('weather_prediction.html', city=city)

@app.route('/api/weather', methods=['POST'])
def get_weather():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400
        
        params = {'units': 'metric', 'appid': OPENWEATHER_API_KEY, 'lang': 'en'}
        geo_params = {}
        
        if 'city' in data:
            params['q'] = geo_params['q'] = data['city'].strip()
        elif 'lat' in data and 'lon' in data:
            params['lat'] = geo_params['lat'] = data['lat']
            params['lon'] = geo_params['lon'] = data['lon']
        else:
            return jsonify({"success": False, "error": "Provide city or coordinates"}), 400
        
        # Get weather data
        weather_response = requests.get(BASE_WEATHER_URL, params=params)
        weather_response.raise_for_status()
        weather_data = weather_response.json()
        
        # Get coordinates if we only had city name
        if 'city' in data:
            geo_response = requests.get(
                "http://api.openweathermap.org/geo/1.0/direct",
                params={'q': data['city'], 'limit': 1, 'appid': OPENWEATHER_API_KEY}
            )
            geo_response.raise_for_status()
            geo_data = geo_response.json()
            if not geo_data:
                return jsonify({"success": False, "error": "City not found"}), 404
            geo_params['lat'] = geo_data[0]['lat']
            geo_params['lon'] = geo_data[0]['lon']
        
        # Get air quality data
        aqi_response = requests.get(AIR_QUALITY_URL, params={
            'lat': geo_params['lat'],
            'lon': geo_params['lon'],
            'appid': OPENWEATHER_API_KEY
        })
        
        # Get UV index data
        uv_response = requests.get(UV_INDEX_URL, params={
            'lat': geo_params['lat'],
            'lon': geo_params['lon'],
            'appid': OPENWEATHER_API_KEY
        })
        
        # Get forecast data for speech and daily forecast
        forecast_response = requests.get(FORECAST_URL, params={
            'lat': geo_params['lat'],
            'lon': geo_params['lon'],
            'appid': OPENWEATHER_API_KEY,
            'units': 'metric',
            'cnt': 40,  # Enough for 5 days (8 intervals/day)
            'lang': 'en'
        })
        
        # Process additional data
        aqi_data = aqi_response.json() if aqi_response.status_code == 200 else None
        uv_data = uv_response.json() if uv_response.status_code == 200 else None
        forecast_data = forecast_response.json() if forecast_response.status_code == 200 else None
        
        # Process hourly forecast
        hourly_forecast = []
        if forecast_data and 'list' in forecast_data:
            for item in forecast_data['list'][:8]:  # Next 24 hours (8 intervals)
                hourly_forecast.append({
                    'time': datetime.fromtimestamp(item['dt']).strftime('%H:%M'),
                    'temp': item['main']['temp'],
                    'description': item['weather'][0]['description'],
                    'icon': item['weather'][0]['icon']
                })
        
        # Process daily forecast (5 days)
        daily_forecast = []
        if forecast_data and 'list' in forecast_data:
            daily_data = {}
            for item in forecast_data['list']:
                date = datetime.fromtimestamp(item['dt']).strftime('%Y-%m-%d')
                if date not in daily_data:
                    daily_data[date] = {
                        'temps': [],
                        'humidity': [],
                        'wind_speed': [],
                        'precipitation': 0,
                        'icon': item['weather'][0]['icon']
                    }
                daily_data[date]['temps'].append(item['main']['temp'])
                daily_data[date]['humidity'].append(item['main']['humidity'])
                daily_data[date]['wind_speed'].append(item['wind']['speed'])
                daily_data[date]['precipitation'] += item.get('rain', {}).get('3h', 0)
            
            for i, (date, data) in enumerate(list(daily_data.items())[:5]):
                day_name = (datetime.today() + timedelta(days=i)).strftime('%a')
                daily_forecast.append({
                    'day': day_name,
                    'high_temp': max(data['temps']),
                    'low_temp': min(data['temps']),
                    'humidity': round(sum(data['humidity']) / len(data['humidity'])),
                    'windspeed': round(sum(data['wind_speed']) / len(data['wind_speed']) * 3.6, 1),  # Convert m/s to km/h
                    'precipitation': round(data['precipitation'], 1),
                    'icon': data['icon']
                })
        
        # Calculate tonight's precipitation (next 12 hours)
        tonight_precipitation = sum(
            item.get('rain', {}).get('3h', 0) 
            for item in forecast_data['list'][:4]  # Next 12 hours (4 intervals)
        ) if forecast_data else 0
        
        # Process current data
        current_data = {
            'temp': round(weather_data['main']['temp']),
            'feels_like': round(weather_data['main']['feels_like']),
            'condition': weather_data['weather'][0]['main'],
            'description': weather_data['weather'][0]['description'].capitalize(),
            'humidity': weather_data['main']['humidity'],
            'wind_speed': weather_data['wind']['speed'],
            'pressure': weather_data['main']['pressure'],
            'icon': weather_data['weather'][0]['icon'],
            'city': weather_data['name'],
            'country': weather_data['sys']['country']
        }

        # Process precipitation data
        precipitation = []
        for item in forecast_data['list'][:6] if forecast_data else []:
            time = datetime.fromtimestamp(item['dt'])
            features = [current_data['temp'], item['main']['humidity'], item['main']['pressure'], item['wind']['speed'], time.hour]
            api_amount = item.get('rain', {}).get('3h', 0) if 'rain' in item else 0
            ml_amount = predictor.predict(features, 'precipitation') or api_amount
            combined_amount = (api_amount + ml_amount) / 2 if predictor.is_trained else api_amount
            
            precipitation.append({
                'time': time.strftime('%I%p').lstrip('0'),
                'amount': max(0, round(combined_amount, 2)),
                'probability': min(100, item.get('pop', 0) * 100)
            })
        
        tonight_total = sum(item['amount'] for item in precipitation) if precipitation else 0
        precipitation_data = {
            'tonight_total': round(tonight_total, 1),
            'hourly': precipitation,
            'prediction_method': 'combined_api_ml' if predictor.is_trained else 'api'
        }

        # Process temperature data
        temperatures = []
        for item in forecast_data['list'][:6] if forecast_data else []:
            time = datetime.fromtimestamp(item['dt'])
            features = [current_data['temp'], item['main']['humidity'], item['main']['pressure'], item['wind']['speed'], time.hour]
            api_temp = item['main']['temp']
            ml_temp = predictor.predict(features, 'temperature') or api_temp
            combined_temp = (api_temp + ml_temp) / 2 if predictor.is_trained else api_temp
            
            temperatures.append({
                'time': time.strftime('%I%p').lstrip('0'),
                'temp': round(combined_temp, 1),
                'feels_like': round(item['main']['feels_like'], 1)
            })
        temperature_data = {'hourly': temperatures}

        # Process humidity data
        humidities = []
        for item in forecast_data['list'][:6] if forecast_data else []:
            time = datetime.fromtimestamp(item['dt'])
            features = [current_data['temp'], item['main']['humidity'], item['main']['pressure'], item['wind']['speed'], time.hour]
            api_humidity = item['main']['humidity']
            ml_humidity = predictor.predict(features, 'humidity') or api_humidity
            combined_humidity = (api_humidity + ml_humidity) / 2 if predictor.is_trained else api_humidity
            
            humidities.append({
                'time': time.strftime('%I%p').lstrip('0'),
                'humidity': round(combined_humidity, 1)
            })
        humidity_data = {'hourly': humidities}

        # Process windspeed data
        windspeeds = []
        for item in forecast_data['list'][:6] if forecast_data else []:
            time = datetime.fromtimestamp(item['dt'])
            features = [current_data['temp'], item['main']['humidity'], item['main']['pressure'], item['wind']['speed'], time.hour]
            api_windspeed = item['wind']['speed']
            ml_windspeed = predictor.predict(features, 'windspeed') or api_windspeed
            combined_windspeed = (api_windspeed + ml_windspeed) / 2 if predictor.is_trained else api_windspeed
            
            windspeeds.append({
                'time': time.strftime('%I%p').lstrip('0'),
                'speed': round(combined_windspeed * 3.6, 1)  # Convert m/s to km/h
            })
        windspeed_data = {'hourly': windspeeds}

        # Process daily data
        daily = []
        days = {}
        for item in forecast_data['list'] if forecast_data else []:
            date = datetime.fromtimestamp(item['dt']).strftime('%Y-%m-%d')
            day_name = datetime.fromtimestamp(item['dt']).strftime('%a')
            features = [current_data['temp'], item['main']['humidity'], item['main']['pressure'], item['wind']['speed'], datetime.fromtimestamp(item['dt']).hour]
            
            api_precip = item.get('rain', {}).get('3h', 0) if 'rain' in item else 0
            api_temp = item['main']['temp']
            api_humidity = item['main']['humidity']
            api_windspeed = item['wind']['speed']
            
            ml_precip = predictor.predict(features, 'precipitation') or api_precip
            ml_temp = predictor.predict(features, 'temperature') or api_temp
            ml_humidity = predictor.predict(features, 'humidity') or api_humidity
            ml_windspeed = predictor.predict(features, 'windspeed') or api_windspeed
            
            combined_precip = (api_precip + ml_precip) / 2 if predictor.is_trained else api_precip
            combined_temp = (api_temp + ml_temp) / 2 if predictor.is_trained else api_temp
            combined_humidity = (api_humidity + ml_humidity) / 2 if predictor.is_trained else api_humidity
            combined_windspeed = (api_windspeed + ml_windspeed) / 2 if predictor.is_trained else api_windspeed

            if date not in days:
                days[date] = {
                    'day': day_name,
                    'high_temp': combined_temp,
                    'low_temp': combined_temp,
                    'precipitation': combined_precip,
                    'humidity': combined_humidity,
                    'windspeed': combined_windspeed * 3.6,
                    'icon': item['weather'][0]['icon']
                }
            else:
                days[date]['high_temp'] = max(days[date]['high_temp'], combined_temp)
                days[date]['low_temp'] = min(days[date]['low_temp'], combined_temp)
                days[date]['precipitation'] += combined_precip
                days[date]['humidity'] = (days[date]['humidity'] + combined_humidity) / 2
                days[date]['windspeed'] = max(days[date]['windspeed'], combined_windspeed * 3.6)

        result = []
        for date, day_data in list(days.items())[:5]:
            result.append({
                'day': day_data['day'],
                'high_temp': round(day_data['high_temp']),
                'low_temp': round(day_data['low_temp']),
                'precipitation': round(day_data['precipitation'], 1),
                'humidity': round(day_data['humidity']),
                'windspeed': round(day_data['windspeed'], 1),
                'icon': day_data['icon']
            })

        # Save weather data
        stored_weather_data = load_data()
        city = data.get('city', 'London').lower()
        city_data = stored_weather_data.get(city, [])
        current_entry = {
            'timestamp': datetime.now().isoformat(),
            'temp': current_data['temp'],
            'humidity': current_data['humidity'],
            'windspeed': current_data['wind_speed'],
            'pressure': current_data['pressure'],
            'aqi': aqi_data['list'][0]['main']['aqi'] if aqi_data else None,
            'precipitation': precipitation_data['tonight_total']
        }
        city_data.append(current_entry)
        stored_weather_data[city] = city_data[-10:]
        save_data(stored_weather_data)

        # Train models if enough data
        if len(city_data) >= 5:
            X = [[d['temp'], d['humidity'], d['pressure'], d['windspeed'], datetime.fromisoformat(d['timestamp']).hour] for d in city_data[:-1]]
            y_precip = [d['precipitation'] for d in city_data[1:]]
            y_temp = [d['temp'] for d in city_data[1:]]
            y_humidity = [d['humidity'] for d in city_data[1:]]
            y_windspeed = [d['windspeed'] for d in city_data[1:]]
            predictor.train_models(X, y_precip, y_temp, y_humidity, y_windspeed)

        # Prepare comprehensive response
        response_data = {
            "success": True,
            "current": current_data,
            "precipitation": precipitation_data,
            "temperature": temperature_data,
            "humidity": humidity_data,
            "windspeed": windspeed_data,
            "daily": result,
            "name": weather_data.get("name"),
            "main": {
                "temp": weather_data["main"]["temp"],
                "feels_like": weather_data["main"]["feels_like"],
                "humidity": weather_data["main"]["humidity"],
                "temp_min": weather_data["main"]["temp_min"],
                "temp_max": weather_data["main"]["temp_max"],
                "pressure": weather_data["main"]["pressure"]
            },
            "wind": {
                "speed": weather_data["wind"]["speed"],
                "deg": weather_data["wind"].get("deg", 0),
                "gust": weather_data["wind"].get("gust", 0),
                "direction": get_wind_direction(weather_data["wind"].get("deg", 0))
            },
            "weather": weather_data["weather"],
            "visibility": weather_data.get("visibility", 0) / 1000,  # in km
            "air_quality": aqi_data['list'][0]['main']['aqi'] if aqi_data else None,
            "uv_index": uv_data.get('value') if uv_data else None,
            "hourly_forecast": hourly_forecast,
            "sunrise": datetime.fromtimestamp(weather_data['sys']['sunrise']).strftime('%H:%M'),
            "sunset": datetime.fromtimestamp(weather_data['sys']['sunset']).strftime('%H:%M'),
            "pollutants": aqi_data['list'][0]['components'] if aqi_data else None,
            "speech_ready": True
        }
        
        return jsonify(response_data)
        
    except requests.exceptions.HTTPError as e:
        error_msg = "City not found" if e.response.status_code == 404 else "Weather service error"
        return jsonify({"success": False, "error": error_msg}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/generate-weather-speech', methods=['POST'])
def generate_weather_speech():
    """Generate natural language weather description for speech"""
    try:
        data = request.get_json()
        if not data or 'weather_data' not in data:
            return jsonify({"success": False, "error": "Weather data required"}), 400

        weather = data['weather_data']
        
        prompt = f"""
        Create a concise, natural-sounding weather report (1-2 paragraphs) for speech synthesis.
        Use this data: {json.dumps(weather, indent=2)}
        
        Guidelines:
        - Speak in second person ("You can expect...")
        - Include temperature, feels-like, conditions, humidity, wind
        - Mention sunrise/sunset if relevant
        - Include air quality and UV index if available
        - Add 1-2 practical tips based on conditions
        - Sound friendly and conversational
        - Keep it under 150 words
        - Example: "Currently in Paris: 18°C, feels like 16°C with light rain..."
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a weather assistant creating spoken reports."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        speech_text = response.choices[0].message.content.strip()
        
        # Ensure proper sentence structure for speech
        speech_text = speech_text.replace('"', '')  # Remove quotes
        speech_text = speech_text.replace('\n', ' ')  # Make it one paragraph
        
        return jsonify({
            "success": True,
            "speech_text": speech_text
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/update-temp', methods=['POST'])
def update_temperature():
    try:
        data = request.get_json()
        city = data.get('city')
        temp = data.get('temp')
        
        if not city or temp is None:
            return jsonify({"success": False, "error": "City and temperature are required"}), 400

        if city not in temperature_data:
            temperature_data[city] = []

        day = len(temperature_data[city]) + 1
        temperature_data[city].append([day, float(temp)])
        
        # Save to file
        with open('temp_data.json', 'w') as f:
            json.dump(temperature_data, f)
            
        return jsonify({"success": True, "message": "Temperature data updated"})
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/predict-temp', methods=['GET'])
def predict_temperature():
    try:
        city = request.args.get('city')
        if not city:
            return jsonify({"success": False, "error": "City is required"}), 400

        if city not in temperature_data or len(temperature_data[city]) < 3:
            return jsonify({"success": False, "error": "Not enough data for prediction"}), 400

        # Prepare data for prediction
        data = temperature_data[city]
        X = np.array([day[0] for day in data]).reshape(-1, 1)
        y = np.array([day[1] for day in data])

        # Train model
        model = LinearRegression()
        model.fit(X, y)

        # Predict next day
        next_day = len(data) + 1
        predicted_temp = model.predict(np.array([[next_day]]))[0]

        return jsonify({
            "success": True,
            "city": city,
            "predicted_temp": round(float(predicted_temp), 2),
            "days": [day[0] for day in data],
            "temps": [day[1] for day in data]
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/<path:path>')
def serve_static(path):
    file_path = os.path.join(PUBLIC_DIR, path)
    if os.path.exists(file_path) and not os.path.isdir(file_path):
        return send_from_directory(PUBLIC_DIR, path)
    return send_from_directory(PUBLIC_DIR, 'index.html')

@app.route('/get_weather', methods=['POST'])
def get_weather_prediction():
    data = request.get_json()
    city = data.get('city', 'London').lower()
    units = data.get('units', 'metric')
    
    try:
        current_url = f"{BASE_WEATHER_URL}?q={city}&appid={OPENWEATHER_API_KEY}&units={units}"
        current_response = requests.get(current_url).json()
        if current_response.get('cod') == "404":
            return jsonify({'error': 'City not found'}), 404
        
        forecast_url = f"{FORECAST_URL}?q={city}&appid={OPENWEATHER_API_KEY}&units={units}"
        forecast_response = requests.get(forecast_url).json()
        
        aqi_url = f"{AIR_QUALITY_URL}?lat={current_response['coord']['lat']}&lon={current_response['coord']['lon']}&appid={OPENWEATHER_API_KEY}"
        aqi_response = requests.get(aqi_url).json()

        weather_data = {
            'current': process_current_data(current_response),
            'precipitation': process_precipitation_data(forecast_response, current_response),
            'temperature': process_temperature_data(forecast_response, current_response),
            'humidity': process_humidity_data(forecast_response, current_response),
            'windspeed': process_windspeed_data(forecast_response, current_response),
            'daily': process_daily_data(forecast_response, current_response),
            'aqi': aqi_response['list'][0]['main']['aqi']
        }

        all_data = load_data()
        city_data = all_data.get(city, [])
        current_entry = {
            'timestamp': datetime.now().isoformat(),
            'temp': weather_data['current']['temp'],
            'humidity': weather_data['current']['humidity'],
            'windspeed': weather_data['current']['wind_speed'],
            'pressure': weather_data['current']['pressure'],
            'aqi': weather_data['aqi'],
            'precipitation': weather_data['precipitation']['tonight_total']
        }
        city_data.append(current_entry)
        all_data[city] = city_data[-10:]
        save_data(all_data)

        if len(city_data) >= 5:
            X = [[d['temp'], d['humidity'], d['pressure'], d['windspeed'], datetime.fromisoformat(d['timestamp']).hour] for d in city_data[:-1]]
            y_precip = [d['precipitation'] for d in city_data[1:]]
            y_temp = [d['temp'] for d in city_data[1:]]
            y_humidity = [d['humidity'] for d in city_data[1:]]
            y_windspeed = [d['windspeed'] for d in city_data[1:]]
            predictor.train_models(X, y_precip, y_temp, y_humidity, y_windspeed)

        return jsonify(weather_data)
    
    except Exception as e:
        print(f"Error in get_weather: {str(e)}")
        return jsonify({'error': str(e)}), 500

def process_current_data(data):
    return {
        'temp': round(data['main']['temp']),
        'feels_like': round(data['main']['feels_like']),
        'condition': data['weather'][0]['main'],
        'description': data['weather'][0]['description'].capitalize(),
        'humidity': data['main']['humidity'],
        'wind_speed': data['wind']['speed'],
        'pressure': data['main']['pressure'],
        'icon': data['weather'][0]['icon'],
        'city': data['name'],
        'country': data['sys']['country']
    }

def process_precipitation_data(forecast_data, current_data):
    precipitation = []
    for item in forecast_data['list'][:6]:
        time = datetime.fromtimestamp(item['dt'])
        features = [current_data['main']['temp'], item['main']['humidity'], item['main']['pressure'], item['wind']['speed'], time.hour]
        api_amount = item.get('rain', {}).get('3h', 0) if 'rain' in item else 0
        ml_amount = predictor.predict(features, 'precipitation') or api_amount
        combined_amount = (api_amount + ml_amount) / 2 if predictor.is_trained else api_amount
        
        precipitation.append({
            'time': time.strftime('%I%p').lstrip('0'),
            'amount': max(0, round(combined_amount, 2)),
            'probability': min(100, item.get('pop', 0) * 100)
        })
    
    tonight_total = sum(item['amount'] for item in precipitation)
    return {
        'tonight_total': round(tonight_total, 1),
        'hourly': precipitation,
        'prediction_method': 'combined_api_ml' if predictor.is_trained else 'api'
    }

def process_temperature_data(forecast_data, current_data):
    temperatures = []
    for item in forecast_data['list'][:6]:
        time = datetime.fromtimestamp(item['dt'])
        features = [current_data['main']['temp'], item['main']['humidity'], item['main']['pressure'], item['wind']['speed'], time.hour]
        api_temp = item['main']['temp']
        ml_temp = predictor.predict(features, 'temperature') or api_temp
        combined_temp = (api_temp + ml_temp) / 2 if predictor.is_trained else api_temp
        
        temperatures.append({
            'time': time.strftime('%I%p').lstrip('0'),
            'temp': round(combined_temp, 1),
            'feels_like': round(item['main']['feels_like'], 1)
        })
    return {'hourly': temperatures}

def process_humidity_data(forecast_data, current_data):
    humidities = []
    for item in forecast_data['list'][:6]:
        time = datetime.fromtimestamp(item['dt'])
        features = [current_data['main']['temp'], item['main']['humidity'], item['main']['pressure'], item['wind']['speed'], time.hour]
        api_humidity = item['main']['humidity']
        ml_humidity = predictor.predict(features, 'humidity') or api_humidity
        combined_humidity = (api_humidity + ml_humidity) / 2 if predictor.is_trained else api_humidity
        
        humidities.append({
            'time': time.strftime('%I%p').lstrip('0'),
            'humidity': round(combined_humidity, 1)
        })
    return {'hourly': humidities}

def process_windspeed_data(forecast_data, current_data):
    windspeeds = []
    for item in forecast_data['list'][:6]:
        time = datetime.fromtimestamp(item['dt'])
        features = [current_data['main']['temp'], item['main']['humidity'], item['main']['pressure'], item['wind']['speed'], time.hour]
        api_windspeed = item['wind']['speed']
        ml_windspeed = predictor.predict(features, 'windspeed') or api_windspeed
        combined_windspeed = (api_windspeed + ml_windspeed) / 2 if predictor.is_trained else api_windspeed
        
        windspeeds.append({
            'time': time.strftime('%I%p').lstrip('0'),
            'speed': round(combined_windspeed * 3.6, 1)  # Convert m/s to km/h
        })
    return {'hourly': windspeeds}

def process_daily_data(forecast_data, current_data):
    daily = []
    days = {}
    for item in forecast_data['list']:
        date = datetime.fromtimestamp(item['dt']).strftime('%Y-%m-%d')
        day_name = datetime.fromtimestamp(item['dt']).strftime('%a')
        features = [current_data['main']['temp'], item['main']['humidity'], item['main']['pressure'], item['wind']['speed'], datetime.fromtimestamp(item['dt']).hour]
        
        api_precip = item.get('rain', {}).get('3h', 0) if 'rain' in item else 0
        api_temp = item['main']['temp']
        api_humidity = item['main']['humidity']
        api_windspeed = item['wind']['speed']
        
        ml_precip = predictor.predict(features, 'precipitation') or api_precip
        ml_temp = predictor.predict(features, 'temperature') or api_temp
        ml_humidity = predictor.predict(features, 'humidity') or api_humidity
        ml_windspeed = predictor.predict(features, 'windspeed') or api_windspeed
        
        combined_precip = (api_precip + ml_precip) / 2 if predictor.is_trained else api_precip
        combined_temp = (api_temp + ml_temp) / 2 if predictor.is_trained else api_temp
        combined_humidity = (api_humidity + ml_humidity) / 2 if predictor.is_trained else api_humidity
        combined_windspeed = (api_windspeed + ml_windspeed) / 2 if predictor.is_trained else api_windspeed

        if date not in days:
            days[date] = {
                'day': day_name,
                'high_temp': combined_temp,
                'low_temp': combined_temp,
                'precipitation': combined_precip,
                'humidity': combined_humidity,
                'windspeed': combined_windspeed * 3.6,
                'icon': item['weather'][0]['icon']
            }
        else:
            days[date]['high_temp'] = max(days[date]['high_temp'], combined_temp)
            days[date]['low_temp'] = min(days[date]['low_temp'], combined_temp)
            days[date]['precipitation'] += combined_precip
            days[date]['humidity'] = (days[date]['humidity'] + combined_humidity) / 2
            days[date]['windspeed'] = max(days[date]['windspeed'], combined_windspeed * 3.6)

    result = []
    for date, day_data in list(days.items())[:5]:
        result.append({
            'day': day_data['day'],
            'high_temp': round(day_data['high_temp']),
            'low_temp': round(day_data['low_temp']),
            'precipitation': round(day_data['precipitation'], 1),
            'humidity': round(day_data['humidity']),
            'windspeed': round(day_data['windspeed'], 1),
            'icon': day_data['icon']
        })
    return result

@app.route('/update-prediction', methods=['POST'])
def update_prediction():
    data = request.get_json()
    if not data or 'city' not in data or 'type' not in data or 'value' not in data:
        return jsonify({"status": "error", "message": "Missing city, type, or value"}), 400

    city = data['city'].lower()
    pred_type = data['type'].lower()
    value = float(data['value'])

    weather_data = load_data()
    city_data = weather_data.get(city, [])
    if len(city_data) < 2:  # Reduced to 2 for testing
        return jsonify({"status": "error", "message": f"Insufficient data for {pred_type} prediction"}), 400

    last = city_data[-1]
    features = [last['temp'], last['humidity'], last['pressure'], last['windspeed'], datetime.now().hour]
    
    if pred_type == "temperature":
        prediction = predictor.predict(features, 'temperature') or value
        return jsonify({"status": "success", "predicted_temperature": float(prediction), "unit": "°C"})
    elif pred_type == "humidity":
        prediction = predictor.predict(features, 'humidity') or value
        return jsonify({"status": "success", "predicted_humidity": float(prediction), "unit": "%"})
    elif pred_type == "windspeed":
        prediction = predictor.predict(features, 'windspeed') or value
        return jsonify({"status": "success", "predicted_windspeed": float(prediction * 3.6), "unit": "km/h"})
    elif pred_type == "precipitation":
        prediction = predictor.predict(features, 'precipitation') or value
        return jsonify({"status": "success", "predicted_precipitation": float(prediction), "unit": "mm"})
    elif pred_type == "aqi":
        # Simple fallback for AQI (no ML model yet)
        return jsonify({"status": "success", "predicted_aqi": int(value), "unit": ""})
    else:
        return jsonify({"status": "error", "message": "Unsupported prediction type"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)