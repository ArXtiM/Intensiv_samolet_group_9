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


# import streamlit as st
# import pandas as pd
# import joblib  # Изменяем импорт
# from datetime import timedelta
# import matplotlib.pyplot as plt
# import numpy as np

# # Загрузка модели и артефактов
# with open(r'C:\Users\Евгения\Desktop\reinforcement_task\data\scaler1.pkl', 'rb') as f:
#     model_artifact = joblib.load(f)  # Загружаем через joblib

# # Извлекаем отдельные компоненты из артефакта
# rf_model = model_artifact['rf_model']
# catboost_model = model_artifact['catboost_model']
# features_order = model_artifact['features_order']

# # Загрузка исторических данных
# historical_df = pd.read_csv(r'C:\Users\Евгения\Desktop\reinforcement_task\data\historical_data.csv')
# historical_df['dt'] = pd.to_datetime(historical_df['dt'])
# historical_df = historical_df.set_index('dt')

# def generate_predictions(start_date, weeks_ahead=6):
#     current_date = start_date
#     predictions = []
    
#     extended_df = historical_df.copy().sort_index()
    
#     for week in range(weeks_ahead):
#         forecast_date = current_date + timedelta(days=7*week)  # Исправляем на дни
        
#         if forecast_date in extended_df.index:
#             continue
            
#         # Создаем признаки согласно обученной модели
#         features = {
#             'month': forecast_date.month,
#             'quarter': forecast_date.quarter,
#             'year': forecast_date.year,
#         }
        
#         # Рассчитываем лаги с правильными временными интервалами (дни)
#         for lag in [1, 2]:
#             lag_date = forecast_date - timedelta(days=lag)
#             available_dates = extended_df.index[extended_df.index <= lag_date]
            
#             if len(available_dates) == 0:
#                 st.error(f"Нет данных для расчета лага {lag} на дату {forecast_date.strftime('%Y-%m-%d')}")
#                 return None
                
#             closest_date = available_dates.max()
#             features[f'lag_{lag}'] = extended_df.loc[closest_date, 'log_price']  # Используем логарифмированные значения
        
#         # Формируем DataFrame с правильным порядком признаков
#         input_data = pd.DataFrame([features])[features_order]
        
#         # Делаем предсказание как в оригинальном коде
#         rf_pred = rf_model.predict(input_data)[0]
#         cat_pred = catboost_model.predict(input_data)[0]
#         log_pred = (rf_pred + cat_pred) / 2
        
#         # Обратное преобразование из логарифма
#         final_pred = np.exp(log_pred)
        
#         # Обновляем данные
#         extended_df.loc[forecast_date] = {
#             'Цена на арматуру': final_pred,
#             'log_price': log_pred,
#             'lag_1': features['lag_1'],
#             'lag_2': features['lag_2'],
#             'month': features['month'],
#             'quarter': features['quarter'],
#             'year': features['year']
#         }
        
#         predictions.append((forecast_date, final_pred))
    
#     return pd.DataFrame(predictions, columns=['Date', 'Predicted Price'])

# # Остальная часть кода остается без изменений
# def main():
#     st.title('Прогнозирование цен на арматуру')
    
#     # выбор даты
#     last_historical_date = historical_df.index.max().date()
#     min_date = last_historical_date + timedelta(weeks=1)
    
#     input_date = st.date_input(
#         'Выберите начальную дату прогноза (понедельник):',
#         min_value=min_date,
#         max_value=last_historical_date + timedelta(weeks=26), # ставим максимальное ограничение по дате в полгода (во избежании слишком неверных прогнозов на очень дальние временные рамки)
#         value=min_date
#     )
    
#     if input_date.weekday() != 0:
#         st.error("Тендеры проводятся только по понедельникам!")
#         return
    
#     if st.button('Сформировать прогноз'):
#         start_date = pd.to_datetime(input_date)
        
#         # прогнозируем
#         predictions = generate_predictions(start_date)
        
#         if predictions is not None:
#             # фильтруем только прогнозируемый период
#             forecast_df = predictions.set_index('Date')
            
#             # получаем текущую цену
#             predicted_prices = forecast_df['Predicted Price'].tolist()
#             current_price = predicted_prices[0]
            
#             # определяем рекомендацию
#             possible_n = 0
#             for n in predicted_prices:
#                 if current_price <= n:
#                     possible_n += 1
#                 elif current_price > n:
#                     break

#             max_n = possible_n if possible_n != 0 else 1
            
#             # визуализация
#             fig, ax = plt.subplots(figsize=(10, 5))
#             ax.plot(forecast_df.index, forecast_df['Predicted Price'], marker='o', 
#                     linestyle='-', linewidth=2, markersize=8, label='Прогнозная цена')
#             ax.axhline(y=current_price, color='red', linestyle='--', 
#                       label=f'Текущая цена: {current_price:.2f} руб/т')
#             ax.set_xlabel('Дата', fontsize=12)
#             ax.set_ylabel('Цена (руб/т)', fontsize=12)
#             ax.set_title('Прогноз цен на арматуру', fontsize=14, pad=20)
#             ax.legend(prop={'size': 10})
#             plt.xticks(rotation=45)
#             plt.grid(True, linestyle='--', alpha=0.7)
#             plt.tight_layout()
#             st.pyplot(fig)
            
#             # вывод дополнительных аналитических данных
#             st.subheader("Детали прогноза:")
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 st.write("**Даты и цены:**")
#                 st.dataframe(forecast_df.reset_index().style.format({
#                     'Date': lambda x: x.strftime('%Y-%m-%d'),
#                     'Predicted Price': '{:.2f} руб'
#                 }))
            
#             with col2:
#                 st.write("**Статистика:**")
#                 st.metric("Средняя цена", f"{forecast_df.mean().values[0]:.2f} руб")
#                 st.metric("Минимальная цена", f"{forecast_df.min().values[0]:.2f} руб")
#                 st.metric("Максимальная цена", f"{forecast_df.max().values[0]:.2f} руб")
            
#             # вывод рекомендаций
#             st.subheader("Рекомендации:")
#             with st.container():
#                 if max_n == 1:
#                     st.success(f"Оптимальный период закупки: {max_n} неделя")
#                 elif max_n in [2, 3, 4]:
#                     st.success(f"Оптимальный период закупки: {max_n} недели")
#                 elif max_n in [5, 6]:
#                     st.success(f"Оптимальный период закупки: {max_n} недель")



# main()

# import streamlit as st
# import pandas as pd
# import pickle
# from datetime import timedelta
# import matplotlib.pyplot as plt

# # загрузка модели
# with open(r'C:\Users\Евгения\Desktop\reinforcement_task\data\scaler1.pkl', 'rb') as f:
#     model = pickle.load(f)

# # загрузка исторических данных
# historical_df = pd.read_csv(r'C:\Users\Евгения\Desktop\reinforcement_task\data\historical_data.csv')
# historical_df['dt'] = pd.to_datetime(historical_df['dt'])
# historical_df = historical_df.set_index('dt')

# def generate_predictions(start_date, weeks_ahead=6):
#     current_date = start_date
#     predictions = []
    
#     # создаем копию исторических данных и сортируем индекс
#     extended_df = historical_df.copy().sort_index()
    
#     # прогнозируем на 6 недель вперед
#     for week in range(weeks_ahead):
#         forecast_date = current_date + timedelta(weeks=week)
        
#         # пропускаем даты, которые уже есть в исторических данных
#         if forecast_date in extended_df.index:
#             continue
            
#         # создаем признаки
#         features = {
#             'year': forecast_date.year,
#             'month': forecast_date.month,
#             'day': forecast_date.day,
#             'week': forecast_date.isocalendar().week,
#         }
        
#         # рассчитываем лаги с использованием ближайших доступных данных
#         for lag in [1, 2]:
#             lag_date = forecast_date - timedelta(weeks=lag)
#             available_dates = extended_df.index[extended_df.index <= lag_date]
            
#             if len(available_dates) == 0:
#                 st.error(f"Нет данных для расчета лага {lag} на дату {forecast_date.strftime('%Y-%m-%d')}")
#                 return None
                
#             closest_date = available_dates.max()
#             features[f'lag_{lag}'] = extended_df.loc[closest_date, 'Цена на арматуру']
        
#         # прогнозируем цену
#         prediction = model.predict(pd.DataFrame([features]))[0]
        
#         # добавляем прогноз в датафрейм
#         extended_df.loc[forecast_date] = {
#             'Цена на арматуру': prediction,
#             'lag_1': features['lag_1'],
#             'lag_2': features['lag_2']
#         }
        
#         predictions.append((forecast_date, prediction))
    
#     return pd.DataFrame(predictions, columns=['Date', 'Predicted Price'])

# def main():
#     st.title('Прогнозирование цен на арматуру')
    
#     # выбор даты
#     last_historical_date = historical_df.index.max().date()
#     min_date = last_historical_date + timedelta(weeks=1)
    
#     input_date = st.date_input(
#         'Выберите начальную дату прогноза (понедельник):',
#         min_value=min_date,
#         max_value=last_historical_date + timedelta(weeks=26), # ставим максимальное ограничение по дате в полгода (во избежании слишком неверных прогнозов на очень дальние временные рамки)
#         value=min_date
#     )
    
#     if input_date.weekday() != 0:
#         st.error("Тендеры проводятся только по понедельникам!")
#         return
    
#     if st.button('Сформировать прогноз'):
#         start_date = pd.to_datetime(input_date)
        
#         # прогнозируем
#         predictions = generate_predictions(start_date)
        
#         if predictions is not None:
#             # фильтруем только прогнозируемый период
#             forecast_df = predictions.set_index('Date')
            
#             # получаем текущую цену
#             predicted_prices = forecast_df['Predicted Price'].tolist()
#             current_price = predicted_prices[0]
            
#             # определяем рекомендацию
#             possible_n = 0
#             for n in predicted_prices:
#                 if current_price <= n:
#                     possible_n += 1
#                 elif current_price > n:
#                     break

#             max_n = possible_n if possible_n != 0 else 1
            
#             # визуализация
#             fig, ax = plt.subplots(figsize=(10, 5))
#             ax.plot(forecast_df.index, forecast_df['Predicted Price'], marker='o', 
#                     linestyle='-', linewidth=2, markersize=8, label='Прогнозная цена')
#             ax.axhline(y=current_price, color='red', linestyle='--', 
#                       label=f'Текущая цена: {current_price:.2f} руб/т')
#             ax.set_xlabel('Дата', fontsize=12)
#             ax.set_ylabel('Цена (руб/т)', fontsize=12)
#             ax.set_title('Прогноз цен на арматуру', fontsize=14, pad=20)
#             ax.legend(prop={'size': 10})
#             plt.xticks(rotation=45)
#             plt.grid(True, linestyle='--', alpha=0.7)
#             plt.tight_layout()
#             st.pyplot(fig)
            
#             # вывод дополнительных аналитических данных
#             st.subheader("Детали прогноза:")
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 st.write("**Даты и цены:**")
#                 st.dataframe(forecast_df.reset_index().style.format({
#                     'Date': lambda x: x.strftime('%Y-%m-%d'),
#                     'Predicted Price': '{:.2f} руб'
#                 }))
            
#             with col2:
#                 st.write("**Статистика:**")
#                 st.metric("Средняя цена", f"{forecast_df.mean().values[0]:.2f} руб")
#                 st.metric("Минимальная цена", f"{forecast_df.min().values[0]:.2f} руб")
#                 st.metric("Максимальная цена", f"{forecast_df.max().values[0]:.2f} руб")
            
#             # вывод рекомендаций
#             st.subheader("Рекомендации:")
#             with st.container():
#                 if max_n == 1:
#                     st.success(f"Оптимальный период закупки: {max_n} неделя")
#                 elif max_n in [2, 3, 4]:
#                     st.success(f"Оптимальный период закупки: {max_n} недели")
#                 elif max_n in [5, 6]:
#                     st.success(f"Оптимальный период закупки: {max_n} недель")



# main()

# импорт библиотек

# Импорт библиотек
# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# from datetime import datetime, timedelta
# import matplotlib.pyplot as plt

# # Загрузка моделей и метаданных
# @st.cache_resource
# def load_model():
#     try:
#         model_data = joblib.load(r'C:\Users\Евгения\Desktop\reinforcement_task\data\armor_price_model_ensemble.pkl')
#         return (
#             model_data['rf_model'],
#             model_data['catboost_model'],
#             model_data['features_order']
#         )
#     except Exception as e:
#         st.error(f"Ошибка загрузки модели: {str(e)}")
#         return None, None, None

# # Загрузка исторических данных
# @st.cache_data
# def load_historical_data():
#     df = pd.read_csv(r'C:\Users\Евгения\Desktop\reinforcement_task\data\historical_data.csv', parse_dates=['dt'])
#     return df.set_index('dt').sort_index().asfreq('D').dropna()

# # Генерация прогноза
# def generate_predictions(start_date, historical_df, models, features_order, days_ahead=7):
#     rf_model, catboost_model = models
#     predictions = []
    
#     current_date = start_date
#     temp_df = historical_df.copy()
    
#     for _ in range(days_ahead):
#         # Создаем фичи для даты
#         features = {
#             'month': current_date.month,
#             'quarter': (current_date.month-1)//3 + 1,
#             'year': current_date.year,
#             'lag_1': temp_df['log_price'].iloc[-1],
#             'lag_2': temp_df['log_price'].iloc[-2]
#         }
        
#         # Формируем DataFrame
#         features_df = pd.DataFrame([features])[features_order]
        
#         # Прогнозируем
#         log_pred_rf = rf_model.predict(features_df)[0]
#         log_pred_cat = catboost_model.predict(features_df)[0]
#         log_pred = (log_pred_rf + log_pred_cat) / 2
#         pred = np.exp(log_pred)
        
#         # Обновляем данные
#         new_row = {
#             'log_price': log_pred,
#             'lag_1': features['lag_1'],
#             'lag_2': features['lag_2'],
#             'month': current_date.month,
#             'quarter': features['quarter'],
#             'year': current_date.year,
#             'Цена на арматуру': pred
#         }
        
#         temp_df.loc[current_date] = new_row
#         predictions.append((current_date, pred))
#         current_date += timedelta(days=1)
    
#     return pd.DataFrame(predictions, columns=['Дата', 'Прогноз']).set_index('Дата')
# def main():
#     st.title('Прогнозирование цен на арматуру')
#     st.markdown("### Анализ рыночной конъюнктуры сталепрокатной продукции")
    
#     # Загрузка данных
#     rf_model, catboost_model, features_order = load_model()
#     historical_df = load_historical_data()
    
#     if rf_model is None or historical_df is None:
#         return
    
#     # Валидация данных
#     last_date = historical_df.index.max().to_pydatetime().date()
#     min_date = last_date + timedelta(days=1)
    
#     with st.sidebar:
#         st.header("Параметры прогноза")
#         input_date = st.date_input(
#             'Дата начала прогноза:',
#             min_value=min_date,
#             max_value=last_date + timedelta(days=90),
#             value=min_date,
#             key='date_input'  # Уникальный ключ
#         )
        
#         days_ahead = st.slider(
#             'Количество дней прогноза:',
#             min_value=7,
#             max_value=30,
#             value=14,
#             key='days_slider'  # Уникальный ключ
#         )
        
#         # Добавляем уникальный ключ для кнопки
#         if st.button('Сформировать прогноз', key='forecast_button'):
#             input_date_pd = pd.to_datetime(input_date)
#             min_date_pd = pd.to_datetime(min_date)
            
#             if input_date_pd < min_date_pd:
#                 st.error("Дата начала прогноза должна быть после последней исторической дати!")
#                 return
#             else:
#                 with st.spinner('Идет расчет прогноза...'):
#                     st.session_state.predictions = generate_predictions(
#                         pd.to_datetime(input_date),
#                         historical_df,
#                         (rf_model, catboost_model),
#                         features_order,
#                         days_ahead
#                     )

#     # Отображение результатов
#     # Отображение результатов
#     # Отображение результатов
#     if 'predictions' in st.session_state and st.session_state.predictions is not None:
#         # Визуализация
#         fig, ax = plt.subplots(figsize=(12, 6))
        
#         # Получаем даты прогноза
#         start_date = st.session_state.predictions.index.min()
#         end_date = st.session_state.predictions.index.max()
        
#         # Рисуем только прогноз
#         st.session_state.predictions['Прогноз'].plot(
#             ax=ax, 
#             label='Прогноз', 
#             color='#ff7f0e', 
#             linewidth=2, 
#             marker='o',
#             title='Прогноз цен на арматуру'
#         )
        
#         # Настраиваем оси и сетку
#         ax.set_xlim(start_date - timedelta(days=1), end_date + timedelta(days=1))
#         ax.set_xlabel('Дата прогноза', fontsize=12)
#         ax.set_ylabel('Цена, руб/т', fontsize=12)
#         ax.grid(True, linestyle='--', alpha=0.7)
#         ax.legend()
        
#         # Форматирование дат на оси X
#         plt.xticks(
#             pd.date_range(start=start_date, end=end_date, freq='D'),
#             rotation=45,
#             ha='right'
#         )
        
#         # Автоматическое выравнивание и отображение
#         plt.tight_layout()
#         st.pyplot(fig)

# if __name__ == '__main__':
#     main()

# # Импорт библиотек
# import streamlit as st
# import pandas as pd
# import pickle
# from datetime import datetime, timedelta
# import matplotlib.pyplot as plt

# # Загрузка модели и данных
# # Загрузка scaler
# # Загрузка модели
# @st.cache_resource
# def load_model():
#     try:
#         with open(r'C:\Users\Евгения\Desktop\reinforcement_task\data\scaler.pkl', 'rb') as f:
#             model = pickle.load(f)
#         return model
#     except FileNotFoundError:
#         st.error("Файл lgbm_model.pkl не найден. Убедитесь, что файл находится в рабочей директории.")
#         return None
#     except Exception as e:
#         st.error(f"Ошибка загрузки модели: {str(e)}")
#         return None

# # Добавить эту функцию перед функцией main()
# @st.cache_data
# def load_historical_data():
#     df = pd.read_csv(r'C:\Users\Евгения\Desktop\reinforcement_task\data\historical_data.csv', parse_dates=['dt'])
#     return df.set_index('dt').sort_index().dropna()

# # Генерация прогноза
# def generate_predictions(start_date, historical_df, model, weeks_ahead=6):
#     predictions = []
#     current_date = start_date
#     extended_df = historical_df.copy().sort_index()
    
#     for week in range(weeks_ahead):
#         forecast_date = current_date + timedelta(weeks=week)
        
#         if forecast_date in extended_df.index:
#             continue
            
#         # Создаем фичи
#         features = {
#             'year': forecast_date.year,
#             'month': forecast_date.month,
#             'day': forecast_date.day,
#             'week': forecast_date.isocalendar().week,
#         }
        
#         # Рассчитываем лаги
#         for lag in [1, 2]:
#             lag_date = forecast_date - timedelta(weeks=lag)
#             available_dates = extended_df.index[extended_df.index <= lag_date]
            
#             if len(available_dates) == 0:
#                 st.error(f"Нет данных для расчета лага {lag} на {forecast_date.strftime('%Y-%m-%d')}")
#                 return None
                
#             closest_date = available_dates.max()
#             features[f'lag_{lag}'] = extended_df.loc[closest_date, 'Цена на арматуру']
        
#         # Прогнозируем
#         prediction = model.predict(pd.DataFrame([features]))[0]
        
#         # Обновляем данные
#         extended_df.loc[forecast_date] = {
#             'Цена на арматуру': prediction,
#             'lag_1': features['lag_1'],
#             'lag_2': features['lag_2']
#         }
        
#         predictions.append((forecast_date, prediction))
    
#     return pd.DataFrame(predictions, columns=['Дата', 'Прогноз']).set_index('Дата')

# def main():
#     st.title('Прогнозирование цен на арматуру')
#     st.markdown("### Анализ рыночной конъюнктуры сталепрокатной продукции")
    
#     # Загрузка данных
#     model = load_model()
#     historical_df = load_historical_data()
    
#     if model is None or historical_df.empty:
#         return
    
#     # Валидация даты
#     last_date = historical_df.index.max().to_pydatetime().date()
#     min_date = last_date + timedelta(days=1)
    
#     with st.sidebar:
#         st.header("Параметры прогноза")
#         input_date = st.date_input(
#             'Дата начала прогноза (понедельник):',
#             min_value=min_date,
#             max_value=last_date + timedelta(weeks=26),
#             value=min_date
#         )
        
#         if st.button('Сформировать прогноз'):
#             if input_date.weekday() != 0:
#                 st.error("Тендеры проводятся только по понедельникам!")
#                 return
            
#             start_date = pd.to_datetime(input_date)
#             with st.spinner('Идет расчет прогноза...'):
#                 st.session_state.predictions = generate_predictions(
#                     start_date,
#                     historical_df,
#                     model
#                 )

#     # Отображение результатов
#     if 'predictions' in st.session_state and st.session_state.predictions is not None:
#         predictions = st.session_state.predictions
#         current_price = predictions.iloc[0]['Прогноз']
        
#         # Визуализация
#         fig, ax = plt.subplots(figsize=(12, 6))
#         predictions['Прогноз'].plot(
#             ax=ax, 
#             label='Прогноз', 
#             color='#ff7f0e', 
#             linewidth=2, 
#             marker='o',
#             title='Прогноз цен на арматуру'
#         )
        
#         # Добавляем текущую цену
#         ax.axhline(
#             y=current_price, 
#             color='red', 
#             linestyle='--', 
#             label=f'Текущая цена: {current_price:.2f} руб/т'
#         )
        
#         # Настройки графика
#         ax.set_xlabel('Дата прогноза', fontsize=12)
#         ax.set_ylabel('Цена, руб/т', fontsize=12)
#         ax.grid(True, linestyle='--', alpha=0.7)
#         ax.legend()
#         plt.xticks(rotation=45)
#         plt.tight_layout()
#         st.pyplot(fig)
        
#         # Аналитика и рекомендации
#         st.subheader("Анализ прогноза")
        
#         # Расчет рекомендации
#         max_n = 0
#         for price in predictions['Прогноз'][1:]:
#             if current_price <= price:
#                 max_n += 1
#             else:
#                 break
        
#         # Статистика
#         col1, col2 = st.columns(2)
#         with col1:
#             st.markdown("**Основные показатели:**")
#             st.metric("Средняя цена", f"{predictions['Прогноз'].mean():.2f} руб/т")
#             st.metric("Минимальная цена", f"{predictions['Прогноз'].min():.2f} руб/т")
            
#         with col2:
#             st.markdown("**Рекомендации:**")
#             if max_n == 0:
#                 st.warning("Рекомендуется провести закупку в ближайший понедельник")
#             else:
#                 weeks_label = "недель" if max_n > 1 else "неделя"
#                 st.success(f"Оптимальный период закупки: {max_n} {weeks_label}")

# if __name__ == '__main__':
#     main()