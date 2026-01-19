import pandas as pd 
import requests
import os
from pymongo import MongoClient

def extract_data(url):
     API_Key=os.getenv('_API_NINJA_KEY_')
     header={"X-Api-Key":API_Key}
     param={'city':'Karachi'}
     response=requests.get(url,headers=header,params=param)
     data=response.json()
     return data

def transform_data(data):
      
     time=pd.Timestamp.now()
     CO=data['CO']['concentration']
     No2=data['NO2']['concentration']
     O3=data['O3']['concentration']
     Pm10=data['PM10']['concentration']
     Pm2_5=data['PM2.5']['concentration']
     So2=data['SO2']['concentration']
     Aqi=data['overall_aqi']
     title=['time','co','no2','o3','pm10','pm2_5','so2','aqi']
     values=[time,CO,No2,O3,Pm10,Pm2_5,So2,Aqi]

     aqi_df=pd.DataFrame(dict(zip(title,values)),index=[0])
     
     return aqi_df.to_dict(orient="records")[0]

def load_data(data_dict):
     mongo_uri=os.getenv("MONGO_URI")
     client = MongoClient(mongo_uri)
     db=client['aqi_data']
     collection=db['karachi_aqi_etl']
     collection.insert_one(data_dict)
     



extract=extract_data('https://api.api-ninjas.com/v1/airquality')
transformed=transform_data(extract)
load_data(transformed)