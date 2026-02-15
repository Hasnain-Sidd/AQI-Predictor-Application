import openmeteo_requests

import pandas as pd
import requests_cache
from retry_requests import retry
from pymongo import MongoClient


cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

url = "https://air-quality-api.open-meteo.com/v1/air-quality"
params = {
	"latitude": 24.8608,
	"longitude": 67.0104,
	"hourly": ["us_aqi", "us_aqi_pm2_5", "us_aqi_pm10", "us_aqi_nitrogen_dioxide", "us_aqi_carbon_monoxide", "us_aqi_ozone", "us_aqi_sulphur_dioxide"],
	"start_date": "2025-10-01",
	"end_date": "2026-01-18",
}
responses = openmeteo.weather_api(url, params=params)

response = responses[0]
print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation: {response.Elevation()} m asl")
print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

hourly = response.Hourly()
hourly_us_aqi = hourly.Variables(0).ValuesAsNumpy()
hourly_us_aqi_pm2_5 = hourly.Variables(1).ValuesAsNumpy()
hourly_us_aqi_pm10 = hourly.Variables(2).ValuesAsNumpy()
hourly_us_aqi_nitrogen_dioxide = hourly.Variables(3).ValuesAsNumpy()
hourly_us_aqi_carbon_monoxide = hourly.Variables(4).ValuesAsNumpy()
hourly_us_aqi_ozone = hourly.Variables(5).ValuesAsNumpy()
hourly_us_aqi_sulphur_dioxide = hourly.Variables(6).ValuesAsNumpy()

hourly_data = {"time": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
	end =  pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}

hourly_data["co"] = hourly_us_aqi_carbon_monoxide
hourly_data["no2"] = hourly_us_aqi_nitrogen_dioxide
hourly_data["o3"] = hourly_us_aqi_ozone
hourly_data["pm10"] = hourly_us_aqi_pm10
hourly_data["pm2_5"] = hourly_us_aqi_pm2_5
hourly_data["so2"] = hourly_us_aqi_sulphur_dioxide
hourly_data["aqi"] = hourly_us_aqi



hourly_dataframe = pd.DataFrame(data = hourly_data)
print("\nHourly data\n", hourly_dataframe)
client=MongoClient("mongodb+srv://hasnainhissam56_db_user:Hasnain1234#@cluster0.idmsuf2.mongodb.net/?appName=Cluster0")
db=client["aqi_data"]
collection=db["karachi_aqi_etl"]
collection.insert_many(hourly_dataframe.to_dict('records'))
