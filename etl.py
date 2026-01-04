import pandas as pd 
import requests
import sqlalchemy
import os

def extract_data(url):
     API_Key=os.getenv('_API_NINJA_KEY_')
     header={"X-Api-Key":API_Key}
     param={'city':'Karachi'}
     response=requests.get(url,headers=header,params=param)
     data=response.json()
     return data

def transform_data(data):
     CO=data['CO']['concentration']
     No2=data['NO2']['concentration']
     O3=data['O3']['concentration']
     Pm10=data['PM10']['concentration']
     Pm2_5=data['PM2.5']['concentration']
     So2=data['SO2']['concentration']
     Aqi=data['overall_aqi']
     title=['CO','No2','O3','Pm10','Pm2_5','So2','Aqi']
     values=[CO,No2,O3,Pm10,Pm2_5,So2,Aqi]
     aqi_df=pd.DataFrame(dict(zip(title,values)),index=[0])
     return aqi_df.to_dict()

def load_data(data_dict,db_connection):
     df=pd.DataFrame(data_dict)
     engine=sqlalchemy.create_engine(db_connection)
     with engine.connect() as connection:
         df.to_sql("aqi_data",if_exists='append',con=connection,index=False)

extract=extract_data('https://api.api-ninjas.com/v1/airquality')
transformed=transform_data(extract)
load_data(transformed,"mysql+pymysql://root:Has1234#@localhost:3306/test_db")