import streamlit as st
import pandas as pd
import joblib
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np

# Настройка кастомных стилей
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Comic+Neue:wght@700&display=swap');
        
        .stApp {
            background-color: #FFF0F5;
            font-family: 'Comic Neue', cursive;
            color: #000000 !important;
        }
        
        h1, h2, h3 {
            color: #FF69B4 !important;
            font-family: 'Comic Neue', cursive !important;
        }
        
        .stDateInput, .stButton>button {
            border-radius: 15px !important;
            border: 2px solid #FF69B4 !important;
            color: #000000 !important;
        }
        
        .stButton>button {
            background-color: #FFB6C1 !important;
            color: #000000 !important;
            font-weight: bold;
        }
        
        .stDataFrame {
            border: 2px solid #FF69B4 !important;
            border-radius: 15px;
            color: #000000 !important;
        }
        
        .metric {
            background-color: #FFE4E1 !important;
            border: 2px solid #FF69B4 !important;
            border-radius: 15px;
            padding: 10px !important;
            color: #000000 !important;
        }
        
        /* Новые стили для текста */
        .stMetric div, .stMetric label {
            color: #000000 !important;
        }
    </style>
""", unsafe_allow_html=True)


with open(r'C:\Users\Евгения\Desktop\reinforcement_task\data\scaler1.pkl', 'rb') as f:
    model_artifact = joblib.load(f)
    # Извлекаем все компоненты из артефакта
    rf_model = model_artifact['rf_model']
    catboost_model = model_artifact['catboost_model']
    features_order = model_artifact['features_order']  # Добавляем эту строку

historical_df = pd.read_csv(r'C:\Users\Евгения\Desktop\reinforcement_task\data\historical_data.csv')
historical_df['dt'] = pd.to_datetime(historical_df['dt'])
historical_df = historical_df.set_index('dt')

def generate_predictions(start_date, weeks_ahead=6):
    current_date = start_date
    predictions = []
    
    extended_df = historical_df.copy().sort_index()
    
    for week in range(weeks_ahead):
        forecast_date = current_date + timedelta(days=7*week)
        
        if forecast_date in extended_df.index:
            continue
            
        features = {
            'month': forecast_date.month,
            'quarter': forecast_date.quarter,
            'year': forecast_date.year,
        }
        
        for lag in [1, 2]:
            lag_date = forecast_date - timedelta(days=lag)
            available_dates = extended_df.index[extended_df.index <= lag_date]
            
            if len(available_dates) == 0:
                st.error(f"Нет данных для расчета лага {lag} на дату {forecast_date.strftime('%Y-%m-%d')}")
                return None
                
            closest_date = available_dates.max()
            features[f'lag_{lag}'] = extended_df.loc[closest_date, 'log_price']
        
        # Исправленная строка с использованием извлеченной features_order
        input_data = pd.DataFrame([features])[features_order]
        
        # Используем извлеченные модели
        rf_pred = rf_model.predict(input_data)[0]  # Изменено
        cat_pred = catboost_model.predict(input_data)[0]  # Изменено
        
        log_pred = (rf_pred + cat_pred) / 2
        final_pred = np.exp(log_pred)
        
        extended_df.loc[forecast_date] = {
            'Цена на арматуру': final_pred,
            'log_price': log_pred,
            'lag_1': features['lag_1'],
            'lag_2': features['lag_2'],
            'month': features['month'],
            'quarter': features['quarter'],
            'year': features['year']
        }
        
        predictions.append((forecast_date, final_pred))
    
    return pd.DataFrame(predictions, columns=['Date', 'Predicted Price'])

def main():
    st.title('🌈 Прогнозирование цен на арматуру')
    
    # выбор даты
    last_historical_date = historical_df.index.max().date()  # Теперь historical_df доступна
    min_date = last_historical_date + timedelta(weeks=1)
    
    input_date = st.date_input(
        'Выберите начальную дату прогноза (понедельник):',
        min_value=min_date,
        max_value=last_historical_date + timedelta(weeks=26),
        value=min_date
    )
    
    if input_date.weekday() != 0:
        st.error("🚨 Тендеры проводятся только по понедельникам!")
        return
    
    if st.button('✨ Сформировать прогноз'):
        start_date = pd.to_datetime(input_date)
        
        # прогнозируем
        predictions = generate_predictions(start_date)
        
        if predictions is not None:
            # фильтруем только прогнозируемый период
            forecast_df = predictions.set_index('Date')
            
            # получаем текущую цену
            predicted_prices = forecast_df['Predicted Price'].tolist()
            current_price = predicted_prices[0]
            
            # определяем рекомендацию
            possible_n = 0
            for n in predicted_prices:
                if current_price <= n:
                    possible_n += 1
                elif current_price > n:
                    break

            max_n = possible_n if possible_n != 0 else 1
            
            # визуализация с розовыми элементами
            # Визуализация с черным текстом
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(forecast_df.index, forecast_df['Predicted Price'], 
                    marker='o', color='#FF69B4', markersize=12,
                    linestyle='-', linewidth=3, label='Прогнозная цена')
            ax.axhline(y=current_price, color='#FF1493', linestyle='--', 
                    linewidth=2, label=f'Текущая цена: {current_price:.2f} руб/т')

            # Изменение цвета текста на черный
            ax.set_xlabel('Дата', fontsize=12, color='black')  # ← Изменено
            ax.set_ylabel('Цена (руб/т)', fontsize=12, color='black')  # ← Изменено
            ax.set_title('🎀 Прогноз цен на арматуру', fontsize=14, pad=20, color='#FF1493')
            ax.tick_params(colors='black')  # ← Изменено
            
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.7, color='#FFB6C1')
            plt.gcf().set_facecolor('#FFF0F5')
            plt.tight_layout()
            st.pyplot(fig)
            
            # вывод дополнительных аналитических данных
            st.subheader("🌸 Детали прогноза:")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**📅 Даты и цены:**")
                st.dataframe(forecast_df.reset_index().style.format({
                    'Date': lambda x: x.strftime('%Y-%m-%d'),
                    'Predicted Price': '{:.2f} руб'
                }).set_properties(**{
                    'background-color': '#FFE4E1',
                    'color': '#000000',  # ← Изменено на черный
                    'border': '1px solid #FF69B4'
                }))
            
            with col2:
                st.write("**📊 Статистика:**")
                st.metric("Средняя цена", 
                          f"{forecast_df.mean().values[0]:.2f} руб",
                          delta_color="off")
                st.metric("Минимальная цена", 
                          f"{forecast_df.min().values[0]:.2f} руб",
                          delta_color="off")
                st.metric("Максимальная цена", 
                          f"{forecast_df.max().values[0]:.2f} руб",
                          delta_color="off")
            
            # вывод рекомендаций
        st.subheader("💖 Рекомендации:")
        recommendation_style = """
            <div style='
                background-color: #FFB6C1;
                padding: 20px;
                border-radius: 15px;
                border: 2px solid #FF69B4;
                color: #000000;
                margin-bottom: 20px;
            '>
        """

        if max_n == 1:
            text = f"## 🎉 Оптимальный период закупки: {max_n} неделя"
        elif max_n in [2, 3, 4]:
            text = f"## 🎉 Оптимальный период закупки: {max_n} недели"
        else:
            text = f"## 🎉 Оптимальный период закупки: {max_n} недель"

        st.markdown(
            recommendation_style + text + "</div>", 
            unsafe_allow_html=True
        )
main()
