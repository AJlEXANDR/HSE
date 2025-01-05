import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from datetime import datetime

month_to_season = {12: "winter", 1: "winter", 2: "winter",
                   3: "spring", 4: "spring", 5: "spring",
                   6: "summer", 7: "summer", 8: "summer",
                   9: "autumn", 10: "autumn", 11: "autumn"}

def load_data(file):
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['rolling_mean_30d'] = df.groupby('city')['temperature'].transform(lambda x: x.rolling(window=30).mean())
    df_mean_std_agg = df.groupby(['city', 'season'])['temperature'].agg(['mean', 'std']).reset_index()
    df = df.merge(df_mean_std_agg, on=['city', 'season'])
    df['anomaly'] = (df['temperature'] < (df['mean'] - 2 * df['std'])) \
        | (df['temperature'] > (df['mean'] + 2 * df['std']))
    df = df.sort_values(by='timestamp')
    return df

def get_current_temperature(city, api_key):
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city,
        'appid': api_key,
        'units': 'metric'
    }
    response = requests.get(base_url, params=params)
    data = response.json()
    if response.status_code == 200:
        return data['main']['temp']
    elif data.get('cod') == 401:
        raise Exception({"cod": data.get('cod'), 
                         "message": "Invalid API key. Please see https://openweathermap.org/faq#error401 for more info."})
    else:
        raise Exception(f"Error: {data['message']}")

def is_temperature_anomal(city, current_temp, df):
    current_season = month_to_season[datetime.now().month]
    seasonal_data = df[(df['city'] == city) & (df['season'] == current_season)]
    mean_temp = seasonal_data['temperature'].mean()
    std_temp = seasonal_data['temperature'].std()
    if mean_temp - 2 * std_temp <= current_temp <= mean_temp + 2 * std_temp:
        return False
    else:
        return True

def display_statistics(df, city):
    city_info = df[df['city'] == city][['temperature', 'rolling_mean_30d']]
    city_stats = city_info.describe()
    st.subheader(f"Описательная статистика по историческим данным для города: {city}")
    st.write(city_stats)

def display_time_series(df, city):
    city_info = df[df['city'] == city]
    fig_ts = px.scatter(city_info, x='timestamp', y='temperature', color='anomaly',
                     color_discrete_map={True: 'red', False: 'green'},
                     title=f'Временной ряд температур с выделением аномалий для города: {city}',
                     labels={'timestamp': 'Дата', 'temperature': 'Температура в градусах Цельсия'})
    st.plotly_chart(fig_ts)

def display_seasonal_profiles(df, city):
    city_info = df[df['city'] == city]
    fig_sp = px.scatter(city_info, x='timestamp', y='temperature', color='season',
                     color_discrete_map={'summer': 'red', 'autumn': 'orange', 'winter': 'blue', 'spring': 'yellow'},
                     title=f'Сезонный температурный профиль для города: {city}',
                     labels={'timestamp': 'Дата', 'temperature': 'Температура в градусах Цельсия'})
    st.plotly_chart(fig_sp)

st.title("Веб-приложение для анализа температур различных городов мира")

uploaded_file = st.file_uploader("Загрузите файл с историческими данными", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    cities = df['city'].unique()

    selected_city = st.selectbox("Выберите город", cities)
    api_key = st.text_input("Введите API-ключ OpenWeatherMap", type="password")

    if api_key:
        try:
            current_temp = get_current_temperature(selected_city, api_key)
            st.write(f"Текущая температура в {selected_city}: {current_temp} градусов по Цельсию")
            st.write(f"Температура {'аномальна' if is_temperature_anomal(selected_city, current_temp, df) else 'нормальна'} для текущего сезона.")
        except Exception as error:
            st.error(str(error))

    display_statistics(df, selected_city)
    display_time_series(df, selected_city)
    display_seasonal_profiles(df, selected_city)