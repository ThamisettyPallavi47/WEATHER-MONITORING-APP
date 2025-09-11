document.addEventListener('DOMContentLoaded', function () {
    // DOM Elements
    const elements = {
        cityInput: document.getElementById('city-input'),
        searchBtn: document.getElementById('search-btn'),
        cityName: document.getElementById('city-name'),
        currentTemp: document.getElementById('current-temp'),
        feelsLike: document.getElementById('feels-like'),
        currentCondition: document.getElementById('current-condition'),
        weatherIcon: document.getElementById('weather-icon'),
        humidity: document.getElementById('humidity'),
        windSpeed: document.getElementById('wind-speed'),
        pressure: document.getElementById('pressure'),
        tonightPrecipitation: document.getElementById('tonight-precipitation'),
        dailyContainer: document.getElementById('daily-container'),
        basicCity: document.querySelector(".city"),
        basicTemp: document.querySelector(".temp"),
        basicHumidity: document.querySelector(".humidity"),
        basicWind: document.querySelector(".wind"),
        basicAqi: document.querySelector(".aqi-value"),
        alertBox: document.querySelector(".alert-message"),
        predictionText: document.querySelector(".prediction"),
        searchBox: document.querySelector(".search input")
    };

    const apiConfig = {
        key: "9685f38713234d8851954d583ec9de70",
        weatherUrl: "https://api.openweathermap.org/data/2.5/weather?units=metric&q=",
        aqiUrl: "https://api.openweathermap.org/data/2.5/air_pollution?"
    };

    let currentCity = 'London';
    let currentDailyData = null;
    let myChart = null;

    // Notification Setup
    if ("Notification" in window && Notification.permission !== "granted") {
        Notification.requestPermission().then(permission => {
            if (permission === "granted") console.log("Notification permission granted.");
        });
    }

    const showNotification = (title, body) => {
        if (Notification.permission === "granted") {
            new Notification(title, { body, icon: "images/weather-icon.png" });
        }
    };

    // Initial fetch
    fetchWeather(currentCity);

    // Event Listeners
    if (elements.searchBtn && (elements.cityInput || elements.searchBox)) {
        elements.searchBtn.addEventListener('click', handleSearch);
        (elements.cityInput || elements.searchBox)?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') handleSearch();
        });
    }

    function handleSearch() {
        const city = (elements.cityInput || elements.searchBox).value.trim();
        if (city) {
            currentCity = city;
            fetchWeather(city);
        } else {
            alert("Please enter a city name!");
        }
    }

    // Location Button
    const locationBtn = createButton('location-btn', () => {
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                position => getWeatherByCoords(position.coords.latitude, position.coords.longitude),
                error => alert(`Error getting location: ${error.message}`)
            );
        } else {
            alert("Geolocation not supported.");
        }
    });
    document.querySelector('.search')?.appendChild(locationBtn);

    // Voice Button
    const voiceBtn = createButton('voice-btn', () => {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) return alert("Voice recognition not supported.");

        const recognition = new SpeechRecognition();
        recognition.lang = 'en-US';
        recognition.interimResults = false;

        recognition.onstart = () => {
            elements.cityInput.placeholder = "Listening...";
            voiceBtn.style.background = '#ff4444';
        };
        recognition.onresult = event => {
            const transcript = event.results[0][0].transcript;
            elements.cityInput.value = transcript;
            fetchWeather(transcript);
        };
        recognition.onerror = event => {
            console.error('Voice recognition error:', event.error);
            elements.cityInput.placeholder = "Error occurred.";
            voiceBtn.style.background = '#ebfffc';
        };
        recognition.onend = () => {
            elements.cityInput.placeholder = "Enter city name";
            voiceBtn.style.background = '#ebfffc';
        };
        recognition.start();
    });
    document.querySelector('.search')?.appendChild(voiceBtn);

    function createButton(className, onClick) {
        const btn = document.createElement('button');
        btn.className = className;
        btn.style.cssText = `border: 0; outline: 0; background: #ebfffc; border-radius: 50%; width: 65px; height: 65px; cursor: pointer; display: flex; align-items: center; justify-content: center; margin-left: 10px;`;
        btn.addEventListener('click', onClick);
        return btn;
    }

    async function fetchWeather(city) {
        try {
            const [backendData, weatherData, aqiData] = await Promise.all([
                fetch('/get_weather', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ city, units: 'metric' })
                }).then(res => res.json()),
                fetch(`${apiConfig.weatherUrl}${city}&appid=${apiConfig.key}`).then(res => res.json()),
                fetch(`${apiConfig.aqiUrl}lat=${weatherData?.coord.lat}&lon=${weatherData?.coord.lon}&appid=${apiConfig.key}`).then(res => res.json())
            ]);

            if (backendData.error || weatherData.cod === "404") throw new Error("City not found");

            updateUI(backendData, weatherData, aqiData);
            currentDailyData = backendData.daily;
            updateDailyForecast('precipitation');
            fetchPredictions(city, weatherData.main.temp, weatherData.main.humidity, weatherData.wind.speed, aqiData.list[0].main.aqi);
        } catch (error) {
            console.error('Error fetching weather:', error);
            alert('Failed to fetch weather data.');
        }
    }

    async function getWeatherByCoords(lat, lon) {
        try {
            const data = await fetch(`${apiConfig.weatherUrl}lat=${lat}&lon=${lon}&appid=${apiConfig.key}`).then(res => res.json());
            if (data.cod === "404") throw new Error("Location not found");
            elements.cityInput.value = data.name;
            fetchWeather(data.name);
        } catch (err) {
            console.error("Error fetching weather by coords:", err);
            alert("Error fetching weather data for your location.");
        }
    }

    function updateUI(backendData, weatherData, aqiData) {
        const { current, precipitation, temperature, windspeed, humidity } = backendData;

        // Dashboard UI
        if (elements.cityName) elements.cityName.textContent = `${current.city}, ${current.country}`;
        if (elements.currentTemp) elements.currentTemp.textContent = `${current.temp}°`;
        if (elements.feelsLike) elements.feelsLike.textContent = current.feels_like;
        if (elements.currentCondition) elements.currentCondition.textContent = current.description;
        if (elements.humidity) elements.humidity.textContent = current.humidity;
        if (elements.windSpeed) elements.windSpeed.textContent = Math.round(current.wind_speed * 3.6);
        if (elements.pressure) elements.pressure.textContent = current.pressure;
        if (elements.weatherIcon) {
            elements.weatherIcon.src = `https://openweathermap.org/img/wn/${current.icon}@2x.png`;
            elements.weatherIcon.alt = current.description;
        }
        if (elements.tonightPrecipitation) elements.tonightPrecipitation.textContent = `${precipitation?.tonight_total || 0} mm`;

        // Basic UI
        if (elements.basicCity) elements.basicCity.textContent = weatherData.name;
        if (elements.basicTemp) elements.basicTemp.textContent = `${Math.round(weatherData.main.temp)}°C`;
        if (elements.basicHumidity) elements.basicHumidity.textContent = `${weatherData.main.humidity}%`;
        if (elements.basicWind) elements.basicWind.textContent = `${(weatherData.wind.speed * 3.6).toFixed(1)} km/h`;
        if (elements.basicAqi) {
            const aqiLevels = ["Good", "Fair", "Moderate", "Poor", "Very Poor"];
            elements.basicAqi.textContent = aqiLevels[aqiData.list[0].main.aqi - 1];
        }

        // Alerts
        if (elements.alertBox) {
            const aqi = aqiData.list[0].main.aqi;
            const temp = weatherData.main.temp;
            const condition = weatherData.weather[0].main;
            let alertMsg = aqi >= 4 ? "⚠️ Poor Air Quality!" : temp >= 35 ? "🔥 High Temperature!" : condition === "Rain" ? "☔ Rain expected." : "";
            elements.alertBox.textContent = alertMsg;
            if (alertMsg) showNotification("Weather Alert", alertMsg);
        }

        updateTabData('precipitation', precipitation.hourly, (hour) => ({
            time: hour.time,
            value: hour.amount,
            height: Math.min(hour.amount * 10, 100),
            title: `${hour.amount.toFixed(2)} mm`
        }));
        updateTabData('temperature', temperature.hourly, (hour) => ({
            time: hour.time,
            value: hour.temp,
            height: ((hour.temp + 40) / 80) * 100,
            title: `${hour.temp}°C (Feels like ${hour.feels_like}°C)`
        }));
        updateTabData('wind', windspeed.hourly, (hour) => ({
            time: hour.time,
            value: hour.speed,
            height: Math.min(hour.speed * 2, 100),
            title: `${hour.speed.toFixed(1)} km/h`
        }));
        updateTabData('humidity', humidity.hourly, (hour) => ({
            time: hour.time,
            value: hour.humidity,
            height: Math.min(hour.humidity, 100),
            title: `${hour.humidity}%`
        }), humidity.hourly);
    }

    function updateTabData(tabName, data, mapFn, humidityData) {
        const tab = document.getElementById(`${tabName}-tab`);
        if (!tab) return;

        const hoursContainer = tab.querySelector(`#${tabName}-hours`);
        const barsContainer = tab.querySelector(`#${tabName}-bars`);
        if (hoursContainer && barsContainer) {
            hoursContainer.innerHTML = '';
            barsContainer.innerHTML = '';
            data.map(mapFn).forEach(item => {
                hoursContainer.innerHTML += `<div class="${tabName}-hour">${item.time}</div>`;
                barsContainer.innerHTML += `<div class="${tabName}-bar"><div class="bar" style="height: ${item.height}px;" title="${item.title}"></div></div>`;
            });
            if (tabName === 'humidity' && elements.avgHumidity) {
                const avg = humidityData.reduce((sum, h) => sum + h.humidity, 0) / humidityData.length;
                document.getElementById('avg-humidity').textContent = Math.round(avg);
            }
        }
    }

    function updateDailyForecast(factor) {
        if (!dailyContainer || !currentDailyData) return;
        dailyContainer.innerHTML = '';

        let title = '';
        let valueKey = '';
        let unit = '';
        switch (factor) {
            case 'precipitation':
                title = '5-DAY PRECIPITATION FORECAST';
                valueKey = 'precipitation';
                unit = 'mm';
                break;
            case 'temperature':
                title = '5-DAY TEMPERATURE FORECAST';
                valueKey = 'high_temp';
                unit = '°C';
                break;
            case 'humidity':
                title = '5-DAY HUMIDITY FORECAST';
                valueKey = 'humidity';
                unit = '%';
                break;
            case 'wind':
                title = '5-DAY WIND SPEED FORECAST';
                valueKey = 'windspeed';
                unit = 'km/h';
                break;
        }

        document.querySelector('.daily-forecast h3').textContent = title;

        currentDailyData.forEach(day => {
            const dayCard = document.createElement('div');
            dayCard.className = 'day-card';

            const dayName = document.createElement('div');
            dayName.className = 'day-name';
            dayName.textContent = day.day;

            const icon = document.createElement('div');
            icon.className = 'day-icon';
            const iconImg = document.createElement('img');
            iconImg.src = `https://openweathermap.org/img/wn/${day.icon}.png`;
            iconImg.alt = '';
            icon.appendChild(iconImg);

            const valueContainer = document.createElement('div');
            valueContainer.className = 'day-values';
            if (factor === 'temperature') {
                const high = document.createElement('div');
                high.className = 'day-high';
                high.textContent = `${day.high_temp}°`;
                const low = document.createElement('div');
                low.className = 'day-low';
                low.textContent = `${day.low_temp}°`;
                valueContainer.appendChild(high);
                valueContainer.appendChild(low);
            } else {
                const value = document.createElement('div');
                value.className = 'day-value';
                value.textContent = `${day[valueKey]}${unit}`;
                valueContainer.appendChild(value);
            }

            dayCard.appendChild(dayName);
            dayCard.appendChild(icon);
            dayCard.appendChild(valueContainer);
            dailyContainer.appendChild(dayCard);
        });
    }

    async function fetchPredictions(city, temp, humidity, windSpeed, aqi) {
        const predictionTypes = [
            { type: 'temperature', value: temp, key: 'predicted_temperature', unit: '°C' },
            { type: 'humidity', value: humidity, key: 'predicted_humidity', unit: '%' },
            { type: 'windspeed', value: windSpeed, key: 'predicted_windspeed', unit: 'km/h' },
            { type: 'precipitation', value: 0, key: 'predicted_precipitation', unit: 'mm' },
            { type: 'aqi', value: aqi, key: 'predicted_aqi', unit: '' }
        ];

        if (predictionText) predictionText.textContent = "Predictions:\n";
        for (const pred of predictionTypes) {
            try {
                const res = await fetch('http://127.0.0.1:5000/update-prediction', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ city, type: pred.type, value: pred.value })
                });
                const result = await res.json();
                if (result.status === "success" && predictionText) {
                    const predValue = pred.type === 'aqi' ? Math.round(result[pred.key]) : result[pred.key].toFixed(1);
                    predictionText.textContent += `${pred.type.charAt(0).toUpperCase() + pred.type.slice(1)}: ${predValue} ${pred.unit}\n`;
                }
                if (pred.type === 'windspeed' && document.getElementById('windspeedChart')) {
                    updateWindSpeedChart(windSpeed * 3.6, result.predicted_windspeed);
                }
            } catch (err) {
                console.error(`Error fetching ${pred.type} prediction:`, err);
            }
        }
    }

    function updateWindSpeedChart(currentSpeed, predictedSpeed) {
        const ctx = document.getElementById('windspeedChart');
        if (!ctx) return;
        if (window.windChart) window.windChart.destroy();

        window.windChart = new Chart(ctx.getContext('2d'), {
            type: 'bar',
            data: {
                labels: ['Current', 'Predicted'],
                datasets: [{
                    label: 'Wind Speed (km/h)',
                    data: [currentSpeed, predictedSpeed],
                    backgroundColor: ['rgba(54, 162, 235, 0.7)', 'rgba(255, 159, 64, 0.7)'],
                    borderColor: ['rgba(54, 162, 235, 1)', 'rgba(255, 159, 64, 1)'],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: { y: { beginAtZero: true, title: { display: true, text: 'Wind Speed (km/h)' } } },
                plugins: { title: { display: true, text: 'Current vs Predicted Wind Speed' } }
            }
        });
    }

    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            document.querySelectorAll('.forecast-tab').forEach(tab => tab.style.display = 'none');
            const tabId = btn.getAttribute('data-tab') + '-tab';
            document.getElementById(tabId).style.display = 'block';
            updateDailyForecast(btn.getAttribute('data-tab'));
        });
    });
});

//map javascript
// Add event listener for map button
document.getElementById('map-btn')?.addEventListener('click', function() {
    window.location.href = '/map';
});