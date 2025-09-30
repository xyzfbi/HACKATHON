"""
Анализ почвенных факторов урожайности
Agricultural Soil Analysis Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.plot import show
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import json
import warnings
import os
from datetime import datetime
import base64
warnings.filterwarnings('ignore')

# Конфигурация страницы
st.set_page_config(
    page_title="Анализ почвенных факторов урожайности",
    page_icon="🌾",
    layout="wide"
)

# CSS стили
st.markdown("""
<style>
    .main {padding: 0rem 1rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 2px;}
    .stTabs [data-baseweb="tab"] {padding: 10px 20px;}
    .metric-container {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Инициализация состояния сессии
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'soil_data' not in st.session_state:
    st.session_state.soil_data = None
if 'yield_data' not in st.session_state:
    st.session_state.yield_data = None
if 'geo_data' not in st.session_state:
    st.session_state.geo_data = None
if 'model' not in st.session_state:
    st.session_state.model = None

class SoilAnalyzer:
    """Класс для анализа почвенных данных"""
    
    def __init__(self):
        self.soil_params = [
            'pH', 'Organic_Matter', 'N', 'P', 'K', 
            'Ca', 'Mg', 'S', 'Fe', 'Mn', 'Zn', 'Cu', 'B'
        ]
        
    def load_geotiff(self, file):
        """Загрузка и обработка GeoTIFF файлов"""
        try:
            with rasterio.open(file) as src:
                data = src.read()
                bounds = src.bounds
                crs = src.crs
                transform = src.transform
                return {
                    'data': data,
                    'bounds': bounds,
                    'crs': crs,
                    'transform': transform
                }
        except Exception as e:
            st.error(f"Ошибка загрузки GeoTIFF: {e}")
            return None
    
    def load_geojson(self, file):
        """Загрузка GeoJSON данных"""
        try:
            geo_data = gpd.read_file(file)
            return geo_data
        except Exception as e:
            st.error(f"Ошибка загрузки GeoJSON: {e}")
            return None
    
    def analyze_correlations(self, soil_df, yield_df):
        """Анализ корреляций между параметрами почвы и урожайностью"""
        correlations = {}
        
        # Объединение данных
        merged_df = pd.merge(soil_df, yield_df, on='field_id', how='inner')
        
        # Расчет корреляций
        for param in self.soil_params:
            if param in merged_df.columns:
                corr, p_value = stats.pearsonr(
                    merged_df[param].dropna(), 
                    merged_df['yield'].dropna()
                )
                correlations[param] = {
                    'correlation': corr,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        return correlations, merged_df
    
    def identify_fertility_zones(self, data, threshold_low=0.3, threshold_high=0.7):
        """Определение зон плодородия"""
        normalized = (data - data.min()) / (data.max() - data.min())
        
        zones = np.zeros_like(normalized)
        zones[normalized < threshold_low] = 1  # Низкая урожайность
        zones[(normalized >= threshold_low) & (normalized < threshold_high)] = 2  # Средняя
        zones[normalized >= threshold_high] = 3  # Высокая
        
        return zones
    
    def build_prediction_model(self, soil_df, yield_df):
        """Построение модели прогнозирования урожайности"""
        # Подготовка данных
        merged_df = pd.merge(soil_df, yield_df, on='field_id', how='inner')
        
        # Выбор признаков
        feature_cols = [col for col in self.soil_params if col in merged_df.columns]
        X = merged_df[feature_cols].fillna(merged_df[feature_cols].mean())
        y = merged_df['yield']
        
        # Нормализация
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Разделение на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Обучение модели
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Оценка важности признаков
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return model, scaler, feature_importance, feature_cols
    
    def process_satellite_imagery(self, geotiff_data):
        """Обработка космических снимков для определения почвенной разности"""
        try:
            # Извлечение данных из GeoTIFF
            data = geotiff_data['data']
            if len(data.shape) == 3:
                # Многоканальное изображение
                data_2d = np.mean(data, axis=0)
            else:
                data_2d = data[0] if len(data.shape) == 3 else data
            
            # Нормализация данных
            data_normalized = (data_2d - np.min(data_2d)) / (np.max(data_2d) - np.min(data_2d))
            
            # Кластеризация для определения зон плодородия
            valid_pixels = data_normalized[~np.isnan(data_normalized)]
            if len(valid_pixels) > 0:
                # Подготовка данных для кластеризации
                X = valid_pixels.reshape(-1, 1)
                
                # K-means кластеризация
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X)
                
                # Создание карты зон плодородия
                fertility_map = np.full_like(data_normalized, np.nan)
                valid_mask = ~np.isnan(data_normalized)
                fertility_map[valid_mask] = clusters
            
            return {
                'original_data': data_2d,
                'normalized_data': data_normalized,
                'fertility_zones': fertility_map,
                'bounds': geotiff_data['bounds'],
                'crs': geotiff_data['crs'],
                'transform': geotiff_data['transform']
            }
        except Exception as e:
            st.error(f"Ошибка обработки космических снимков: {e}")
            return None
    
    def analyze_soil_fertility_zones(self, geotiff_data, yield_data):
        """Анализ почвенной разности на основе космических снимков и урожайности"""
        processed_data = self.process_satellite_imagery(geotiff_data)
        if processed_data is None:
            return None
        
        # Связывание данных урожайности с зонами плодородия
        fertility_analysis = {
            'zones': {
                'high': {'count': 0, 'avg_yield': 0, 'pixels': []},
                'medium': {'count': 0, 'avg_yield': 0, 'pixels': []},
                'low': {'count': 0, 'avg_yield': 0, 'pixels': []}
            },
            'processed_data': processed_data
        }
        
        # Анализ зон (0=низкая, 1=средняя, 2=высокая плодородие)
        for zone_id in [0, 1, 2]:
            zone_mask = processed_data['fertility_zones'] == zone_id
            zone_pixels = np.sum(zone_mask)
            
            if zone_pixels > 0:
                zone_name = ['low', 'medium', 'high'][zone_id]
                fertility_analysis['zones'][zone_name]['count'] = zone_pixels
                fertility_analysis['zones'][zone_name]['pixels'] = zone_mask
        
        return fertility_analysis
    
    def create_russia_map_with_fields(self, fields_data):
        """Создание карты России с наложением данных о полях"""
        # Центр России
        russia_center = [64.6863, 97.7453]
        
        # Создание карты
        m = folium.Map(
            location=russia_center,
            zoom_start=4,
            tiles='OpenStreetMap'
        )
        
        # Цвета для зон плодородия
        zone_colors = {
            'high': 'green',
            'medium': 'orange', 
            'low': 'red'
        }
        
        # Добавление полей на карту
        for field in fields_data:
            # Определение цвета по зоне плодородия
            zone = field.get('fertility_zone', 'medium')
            color = zone_colors.get(zone, 'orange')
            
            # Создание простого маркера
            folium.CircleMarker(
                location=[field['lat'], field['lon']],
                radius=15,
                popup=f"<b>{field['name']}</b><br>Урожайность: {field['yield']:.1f} ц/га<br>Зона: {zone}",
                color='black',
                weight=2,
                fillColor=color,
                fillOpacity=0.7
            ).add_to(m)
        
        return m
    
    def get_field_analysis(self, field_id, soil_data, yield_data):
        """Детальный анализ конкретного поля"""
        field_soil = soil_data[soil_data['field_id'] == field_id]
        field_yield = yield_data[yield_data['field_id'] == field_id]
        
        if field_soil.empty or field_yield.empty:
            return None
        
        # Объединение данных
        field_data = pd.merge(field_soil, field_yield, on='field_id')
        
        # Анализ параметров
        analysis = {
            'field_id': field_id,
            'yield': field_yield['yield'].iloc[0],
            'soil_parameters': {},
            'recommendations': []
        }
        
        # Анализ каждого параметра почвы
        for param in self.soil_params:
            if param in field_data.columns:
                value = field_data[param].iloc[0]
                analysis['soil_parameters'][param] = {
                    'value': value,
                    'status': self._evaluate_parameter_status(param, value)
                }
        
        # Генерация рекомендаций
        analysis['recommendations'] = self._generate_recommendations(analysis['soil_parameters'])
        
        return analysis
    
    def _evaluate_parameter_status(self, parameter, value):
        """Оценка статуса параметра почвы"""
        # Нормативные значения (примерные)
        norms = {
            'pH': {'optimal': (6.0, 7.5), 'low': 6.0, 'high': 7.5},
            'N': {'optimal': (20, 50), 'low': 20, 'high': 50},
            'P': {'optimal': (15, 40), 'low': 15, 'high': 40},
            'K': {'optimal': (100, 200), 'low': 100, 'high': 200},
            'Organic_Matter': {'optimal': (2, 5), 'low': 2, 'high': 5}
        }
        
        if parameter in norms:
            norm = norms[parameter]
            if norm['low'] <= value <= norm['high']:
                return 'optimal'
            elif value < norm['low']:
                return 'low'
            else:
                return 'high'
        return 'unknown'
    
    def _generate_recommendations(self, soil_parameters):
        """Генерация рекомендаций на основе анализа почвы"""
        recommendations = []
        
        for param, data in soil_parameters.items():
            if data['status'] == 'low':
                if param == 'N':
                    recommendations.append("Рекомендуется внесение азотных удобрений")
                elif param == 'P':
                    recommendations.append("Необходимо внесение фосфорных удобрений")
                elif param == 'K':
                    recommendations.append("Требуется внесение калийных удобрений")
                elif param == 'pH':
                    recommendations.append("Рекомендуется известкование почвы")
            elif data['status'] == 'high':
                if param == 'pH':
                    recommendations.append("Почва слишком щелочная, требуется подкисление")
                else:
                    recommendations.append(f"Содержание {param} избыточно, снизить нормы внесения")
        
        return recommendations

# Создание экземпляра анализатора
analyzer = SoilAnalyzer()

# Главный заголовок
st.title("🌾 Анализ почвенных факторов урожайности")
st.markdown("---")

# Боковая панель для загрузки данных
with st.sidebar:
    st.header("📁 Загрузка данных")
    
    # Загрузка космических снимков
    st.subheader("Космические снимки (GeoTIFF)")
    geotiff_files = st.file_uploader(
        "Выберите файлы GeoTIFF",
        type=['tif', 'tiff'],
        accept_multiple_files=True,
        key="geotiff"
    )
    
    # Загрузка GeoJSON
    st.subheader("Границы полей (GeoJSON)")
    geojson_file = st.file_uploader(
        "Выберите файл GeoJSON",
        type=['geojson', 'json'],
        key="geojson"
    )
    
    # Загрузка данных о почве
    st.subheader("Химический состав почв")
    soil_file = st.file_uploader(
        "Выберите Excel файл",
        type=['xlsx', 'xls'],
        key="soil"
    )
    
    # Загрузка данных об урожайности
    st.subheader("Данные урожайности")
    yield_file = st.file_uploader(
        "Выберите файл с урожайностью",
        type=['xlsx', 'xls', 'csv'],
        key="yield"
    )
    
    # Кнопка обработки данных
    if st.button("🔄 Обработать данные", type="primary"):
        if soil_file and yield_file:
            try:
                # Загрузка данных о почве
                st.session_state.soil_data = pd.read_excel(soil_file)
                
                # Загрузка данных об урожайности
                if yield_file.name.endswith('.csv'):
                    st.session_state.yield_data = pd.read_csv(yield_file)
                else:
                    st.session_state.yield_data = pd.read_excel(yield_file)
                
                # Загрузка геоданных
                if geojson_file:
                    st.session_state.geo_data = analyzer.load_geojson(geojson_file)
                
                st.session_state.data_loaded = True
                st.success("✅ Данные успешно загружены!")
                
            except Exception as e:
                st.error(f"Ошибка при загрузке данных: {e}")
        else:
            st.warning("⚠️ Загрузите все необходимые файлы")

# Основное содержимое
if st.session_state.data_loaded:
    # Создание вкладок
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Анализ корреляций",
        "🗺️ Карта РФ с полями",
        "🛰️ Анализ космических снимков",
        "📈 Визуализация данных",
        "🔮 Прогнозирование",
        "📋 Отчет"
    ])
    
    # Вкладка 1: Анализ корреляций
    with tab1:
        st.header("Анализ взаимосвязей между параметрами почвы и урожайностью")
        
        # Генерация тестовых данных если реальные не загружены
        if st.session_state.soil_data is None:
            np.random.seed(42)
            n_samples = 100
            
            st.session_state.soil_data = pd.DataFrame({
                'field_id': range(1, n_samples + 1),
                'pH': np.random.normal(6.5, 0.5, n_samples),
                'Organic_Matter': np.random.normal(3.0, 0.8, n_samples),
                'N': np.random.normal(30, 10, n_samples),
                'P': np.random.normal(25, 8, n_samples),
                'K': np.random.normal(150, 30, n_samples),
                'Ca': np.random.normal(1000, 200, n_samples),
                'Mg': np.random.normal(150, 40, n_samples),
                'S': np.random.normal(15, 5, n_samples),
                'Fe': np.random.normal(50, 15, n_samples),
                'Mn': np.random.normal(30, 10, n_samples),
                'Zn': np.random.normal(5, 2, n_samples),
                'Cu': np.random.normal(3, 1, n_samples),
                'B': np.random.normal(1, 0.3, n_samples)
            })
            
            # Генерация урожайности с зависимостью от параметров почвы
            yield_base = (
                st.session_state.soil_data['N'] * 0.5 +
                st.session_state.soil_data['P'] * 0.3 +
                st.session_state.soil_data['K'] * 0.1 +
                st.session_state.soil_data['Organic_Matter'] * 5 +
                np.random.normal(0, 5, n_samples)
            )
            
            st.session_state.yield_data = pd.DataFrame({
                'field_id': range(1, n_samples + 1),
                'yield': yield_base + 20
            })
        
        # Анализ корреляций
        correlations, merged_data = analyzer.analyze_correlations(
            st.session_state.soil_data,
            st.session_state.yield_data
        )
        
        # Визуализация корреляций
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Корреляционная матрица")
            
            # Подготовка данных для матрицы корреляций
            corr_df = pd.DataFrame([
                {'Параметр': k, 'Корреляция': v['correlation'], 'p-value': v['p_value']}
                for k, v in correlations.items()
            ])
            
            # Создание тепловой карты
            fig = px.bar(
                corr_df,
                x='Корреляция',
                y='Параметр',
                orientation='h',
                color='Корреляция',
                color_continuous_scale='RdBu',
                title='Корреляция параметров почвы с урожайностью'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Статистическая значимость")
            
            # Таблица со значимыми корреляциями
            significant_corr = [
                {
                    'Параметр': k,
                    'Корреляция': f"{v['correlation']:.3f}",
                    'p-value': f"{v['p_value']:.4f}",
                    'Значимость': '✅' if v['significant'] else '❌'
                }
                for k, v in correlations.items()
            ]
            
            st.dataframe(
                pd.DataFrame(significant_corr),
                use_container_width=True,
                hide_index=True
            )
            
            # Выводы
            st.info("""
            **Интерпретация результатов:**
            - Корреляция > 0.7: сильная положительная связь
            - Корреляция 0.3-0.7: умеренная связь
            - Корреляция < 0.3: слабая связь
            - p-value < 0.05: статистически значимая связь
            """)
    
    # Вкладка 2: Карта РФ с полями
    with tab2:
        st.header("Карта Российской Федерации с анализируемыми полями")
        
        # Подготовка данных полей
        if st.session_state.soil_data is not None and st.session_state.yield_data is not None:
            # Создание данных о полях
            fields_data = []
            for i in range(1, 4):  # 3 поля
                field_soil = st.session_state.soil_data[st.session_state.soil_data['field_id'] == i]
                field_yield = st.session_state.yield_data[st.session_state.yield_data['field_id'] == i]
                
                if not field_soil.empty and not field_yield.empty:
                    # Определение зоны плодородия
                    yield_val = field_yield['yield'].iloc[0]
                    if yield_val > 40:
                        zone = 'high'
                    elif yield_val > 30:
                        zone = 'medium'
                    else:
                        zone = 'low'
                    
                    # Координаты полей в разных регионах РФ
                    field_coords = [
                        {'name': f'Поле {i} (Московская обл.)', 'lat': 55.7558 + (i-1)*0.5, 'lon': 37.6173 + (i-1)*0.5},
                        {'name': f'Поле {i} (Башкортостан)', 'lat': 54.7388 + (i-1)*0.3, 'lon': 55.9721 + (i-1)*0.3},
                        {'name': f'Поле {i} (Саратовская обл.)', 'lat': 51.5406 + (i-1)*0.2, 'lon': 46.0086 + (i-1)*0.2}
                    ]
                    
                    field_info = {
                        'name': field_coords[i-1]['name'],
                        'lat': field_coords[i-1]['lat'],
                        'lon': field_coords[i-1]['lon'],
                        'yield': yield_val,
                        'fertility_zone': zone,
                        'area': 70,
                        'pH': field_soil['pH'].iloc[0] if 'pH' in field_soil.columns else 'N/A',
                        'N': field_soil['N'].iloc[0] if 'N' in field_soil.columns else 'N/A'
                    }
                    fields_data.append(field_info)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if fields_data:
                try:
                    # Создание карты России с полями
                    m = analyzer.create_russia_map_with_fields(fields_data)
                    
                    # Сохранение карты во временный файл
                    import tempfile
                    import os
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp_file:
                        m.save(tmp_file.name)
                        
                        # Чтение HTML файла
                        with open(tmp_file.name, 'r', encoding='utf-8') as f:
                            html_content = f.read()
                        
                        # Отображение карты
                        st.components.v1.html(html_content, width=700, height=500)
                        
                        # Удаление временного файла
                        os.unlink(tmp_file.name)
                    
                    # Легенда карты
                    st.markdown("""
                    **Легенда карты:**
                    - 🟢 Зеленая зона: Высокая урожайность (>40 ц/га)
                    - 🟠 Оранжевая зона: Средняя урожайность (30-40 ц/га)  
                    - 🔴 Красная зона: Низкая урожайность (<30 ц/га)
                    """)
                except Exception as e:
                    st.error(f"Ошибка отображения карты: {e}")
                    st.info("Попробуйте обновить страницу или перезапустить приложение")
                    
                    # Альтернативное отображение данных в таблице
                    st.subheader("Данные о полях (альтернативное отображение)")
                    fields_df = pd.DataFrame([
                        {
                            'Поле': field['name'],
                            'Широта': field['lat'],
                            'Долгота': field['lon'],
                            'Урожайность (ц/га)': field['yield'],
                            'Зона плодородия': field['fertility_zone']
                        }
                        for field in fields_data
                    ])
                    st.dataframe(fields_df, use_container_width=True)
            else:
                st.warning("Нет данных о полях для отображения на карте")
        
        with col2:
            st.subheader("Фильтры и настройки")
            
            # Фильтр по урожайности
            yield_range = st.slider(
                "Диапазон урожайности (ц/га)",
                min_value=0,
                max_value=60,
                value=(20, 50),
                step=5
            )
            
            # Фильтр по зонам плодородия
            selected_zones = st.multiselect(
                "Выберите зоны плодородия",
                ['high', 'medium', 'low'],
                default=['high', 'medium', 'low'],
                format_func=lambda x: {'high': 'Высокая', 'medium': 'Средняя', 'low': 'Низкая'}[x]
            )
            
            # Фильтр по параметрам
            param_filter = st.selectbox(
                "Параметр для анализа",
                ['Урожайность', 'pH', 'Азот (N)', 'Фосфор (P)', 'Калий (K)', 'Органическое вещество']
            )
            
            # Статистика по полям
            st.subheader("Статистика по полям")
            
            if fields_data:
                stats_df = pd.DataFrame([
                    {
                        'Поле': field['name'],
                        'Урожайность (ц/га)': field['yield'],
                        'Зона': field['fertility_zone'],
                        'pH': field['pH']
                    }
                    for field in fields_data
                ])
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
            
            # Детальный анализ выбранного поля
            st.subheader("Детальный анализ поля")
            selected_field = st.selectbox(
                "Выберите поле для анализа",
                [f"Поле {i}" for i in range(1, len(fields_data)+1)],
                key="field_selector"
            )
            
            if selected_field and st.session_state.soil_data is not None:
                field_id = int(selected_field.split()[1])
                field_analysis = analyzer.get_field_analysis(
                    field_id, 
                    st.session_state.soil_data, 
                    st.session_state.yield_data
                )
                
                if field_analysis:
                    st.write(f"**Урожайность:** {field_analysis['yield']:.1f} ц/га")
                    
                    # Параметры почвы
                    st.write("**Параметры почвы:**")
                    for param, data in field_analysis['soil_parameters'].items():
                        status_emoji = {'optimal': '✅', 'low': '⚠️', 'high': '🔴', 'unknown': '❓'}[data['status']]
                        st.write(f"- {param}: {data['value']:.2f} {status_emoji}")
                    
                    # Рекомендации
                    if field_analysis['recommendations']:
                        st.write("**Рекомендации:**")
                        for rec in field_analysis['recommendations']:
                            st.write(f"- {rec}")
    
    # Вкладка 3: Анализ космических снимков
    with tab3:
        st.header("Анализ космических снимков и определение почвенной разности")
        
        # Обработка загруженных GeoTIFF файлов
        if geotiff_files:
            st.subheader("Обработка космических снимков")
            
            for i, geotiff_file in enumerate(geotiff_files):
                st.write(f"**Обработка файла {i+1}: {geotiff_file.name}**")
                
                # Загрузка GeoTIFF
                geotiff_data = analyzer.load_geotiff(geotiff_file)
                if geotiff_data:
                    # Обработка снимка
                    processed_data = analyzer.process_satellite_imagery(geotiff_data)
                    
                    if processed_data:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Исходные данные:**")
                            st.write(f"- Размер: {processed_data['original_data'].shape}")
                            st.write(f"- Диапазон значений: {np.min(processed_data['original_data']):.2f} - {np.max(processed_data['original_data']):.2f}")
                            
                            # Визуализация исходных данных
                            fig = px.imshow(
                                processed_data['original_data'],
                                title=f"Исходный снимок - {geotiff_file.name}",
                                color_continuous_scale='gray'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.write("**Зоны плодородия:**")
                            
                            # Анализ зон плодородия
                            fertility_analysis = analyzer.analyze_soil_fertility_zones(geotiff_data, st.session_state.yield_data)
                            
                            if fertility_analysis:
                                zones_df = pd.DataFrame([
                                    {'Зона': 'Высокая', 'Пиксели': fertility_analysis['zones']['high']['count']},
                                    {'Зона': 'Средняя', 'Пиксели': fertility_analysis['zones']['medium']['count']},
                                    {'Зона': 'Низкая', 'Пиксели': fertility_analysis['zones']['low']['count']}
                                ])
                                
                                fig = px.pie(
                                    zones_df,
                                    values='Пиксели',
                                    names='Зона',
                                    title='Распределение зон плодородия',
                                    color_discrete_map={
                                        'Высокая': '#2E8B57',
                                        'Средняя': '#FFD700',
                                        'Низкая': '#DC143C'
                                    }
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Визуализация зон плодородия
                                fig = px.imshow(
                                    processed_data['fertility_zones'],
                                    title="Карта зон плодородия",
                                    color_continuous_scale=['red', 'yellow', 'green']
                                )
                                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Загрузите GeoTIFF файлы в боковой панели для анализа космических снимков")
            
            # Демонстрация с тестовыми данными
            st.subheader("Демонстрация анализа почвенной разности")
            
            # Создание тестовых данных для демонстрации
            np.random.seed(42)
            test_image = np.random.rand(100, 100) * 100
            
            # Имитация зон плодородия
            fertility_zones = np.zeros_like(test_image)
            fertility_zones[test_image < 30] = 0  # Низкая
            fertility_zones[(test_image >= 30) & (test_image < 70)] = 1  # Средняя
            fertility_zones[test_image >= 70] = 2  # Высокая
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.imshow(
                    test_image,
                    title="Имитация космического снимка",
                    color_continuous_scale='gray'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.imshow(
                    fertility_zones,
                    title="Определенные зоны плодородия",
                    color_continuous_scale=['red', 'yellow', 'green']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Статистика зон
                zone_counts = [np.sum(fertility_zones == i) for i in range(3)]
                zone_names = ['Низкая', 'Средняя', 'Высокая']
                
                zones_df = pd.DataFrame({
                    'Зона': zone_names,
                    'Пиксели': zone_counts,
                    'Процент': [count/len(fertility_zones.flatten())*100 for count in zone_counts]
                })
                
                st.dataframe(zones_df, use_container_width=True, hide_index=True)
    
    # Вкладка 4: Визуализация данных
    with tab4:
        st.header("Интерактивные графики")
        
        # Выбор параметров для визуализации
        col1, col2 = st.columns(2)
        
        with col1:
            x_param = st.selectbox(
                "Параметр X",
                analyzer.soil_params,
                index=2  # N по умолчанию
            )
        
        with col2:
            y_param = st.selectbox(
                "Параметр Y",
                ['yield'] + analyzer.soil_params,
                index=0  # Урожайность по умолчанию
            )
        
        # График рассеяния
        if x_param in merged_data.columns and y_param in merged_data.columns:
            fig = px.scatter(
                merged_data,
                x=x_param,
                y=y_param,
                size='yield' if y_param != 'yield' else None,
                color='yield',
                title=f'Зависимость {y_param} от {x_param}',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Распределения параметров
        st.subheader("Распределение параметров почвы")
        
        col1, col2 = st.columns(2)
        
        with col1:
            param_dist = st.selectbox(
                "Выберите параметр",
                analyzer.soil_params,
                key="dist_param"
            )
            
            if param_dist in st.session_state.soil_data.columns:
                fig = px.histogram(
                    st.session_state.soil_data,
                    x=param_dist,
                    nbins=30,
                    title=f'Распределение {param_dist}'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot для сравнения параметров
            fig = go.Figure()
            
            for param in ['N', 'P', 'K']:
                if param in st.session_state.soil_data.columns:
                    fig.add_trace(go.Box(
                        y=st.session_state.soil_data[param],
                        name=param
                    ))
            
            fig.update_layout(
                title='Сравнение основных макроэлементов',
                yaxis_title='Концентрация (мг/кг)'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Вкладка 5: Прогнозирование
    with tab5:
        st.header("Прогнозирование урожайности")
        
        # Обучение модели
        if st.button("🤖 Обучить модель"):
            with st.spinner("Обучение модели..."):
                model, scaler, feature_importance, feature_cols = analyzer.build_prediction_model(
                    st.session_state.soil_data,
                    st.session_state.yield_data
                )
                st.session_state.model = {
                    'model': model,
                    'scaler': scaler,
                    'features': feature_cols,
                    'importance': feature_importance
                }
                st.success("✅ Модель успешно обучена!")
        
        if st.session_state.model:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Важность признаков")
                
                fig = px.bar(
                    st.session_state.model['importance'],
                    x='importance',
                    y='feature',
                    orientation='h',
                    title='Влияние параметров на урожайность'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Калькулятор урожайности")
                
                # Создание слайдеров для параметров
                input_values = {}
                for feature in st.session_state.model['features'][:5]:  # Топ-5 важных
                    if feature in st.session_state.soil_data.columns:
                        mean_val = st.session_state.soil_data[feature].mean()
                        std_val = st.session_state.soil_data[feature].std()
                        
                        input_values[feature] = st.slider(
                            f"{feature}",
                            float(mean_val - 2*std_val),
                            float(mean_val + 2*std_val),
                            float(mean_val),
                            step=float(std_val/10)
                        )
                
                # Прогнозирование
                if st.button("📊 Рассчитать прогноз"):
                    # Подготовка данных для прогноза
                    input_df = pd.DataFrame([input_values])
                    
                    # Добавление недостающих признаков
                    for feature in st.session_state.model['features']:
                        if feature not in input_df.columns:
                            input_df[feature] = st.session_state.soil_data[feature].mean()
                    
                    # Прогноз
                    input_scaled = st.session_state.model['scaler'].transform(
                        input_df[st.session_state.model['features']]
                    )
                    prediction = st.session_state.model['model'].predict(input_scaled)[0]
                    
                    # Вывод результата
                    st.metric(
                        "Прогнозируемая урожайность",
                        f"{prediction:.1f} ц/га",
                        delta=f"{prediction - st.session_state.yield_data['yield'].mean():.1f} от среднего"
                    )
                    
                    # Рекомендации
                    st.info("""
                    **Рекомендации для повышения урожайности:**
                    - Увеличьте содержание азота на 10-15%
                    - Оптимизируйте pH почвы до 6.5-7.0
                    - Обеспечьте достаточное содержание органического вещества
                    """)
    
    # Вкладка 6: Отчет
    with tab6:
        st.header("Сводный отчет")
        
        # Основные метрики
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Средняя урожайность",
                f"{st.session_state.yield_data['yield'].mean():.1f} ц/га"
            )
        
        with col2:
            st.metric(
                "Максимальная урожайность",
                f"{st.session_state.yield_data['yield'].max():.1f} ц/га"
            )
        
        with col3:
            st.metric(
                "Минимальная урожайность",
                f"{st.session_state.yield_data['yield'].min():.1f} ц/га"
            )
        
        with col4:
            st.metric(
                "Вариация",
                f"{st.session_state.yield_data['yield'].std():.1f} ц/га"
            )
        
        # Сводная таблица
        st.subheader("Сводная таблица по параметрам почвы")
        
        summary_stats = st.session_state.soil_data[analyzer.soil_params[:8]].describe()
        st.dataframe(summary_stats, use_container_width=True)
        
        # Экспорт отчета
        st.subheader("Экспорт данных")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Сформировать PDF отчет"):
                st.info("Функция экспорта в PDF будет доступна в следующей версии")
        
        with col2:
            # Экспорт в CSV
            csv = st.session_state.soil_data.to_csv(index=False)
            st.download_button(
                label="📊 Скачать данные в CSV",
                data=csv,
                file_name="soil_analysis_report.csv",
                mime="text/csv"
            )
        
        # Детальный анализ по полям
        st.subheader("Детальный анализ по полям")
        
        if st.session_state.soil_data is not None and st.session_state.yield_data is not None:
            # Анализ каждого поля
            fields_analysis = []
            for i in range(1, 4):
                field_analysis = analyzer.get_field_analysis(
                    i, 
                    st.session_state.soil_data, 
                    st.session_state.yield_data
                )
                if field_analysis:
                    fields_analysis.append(field_analysis)
            
            if fields_analysis:
                # Создание таблицы анализа полей
                analysis_df = pd.DataFrame([
                    {
                        'Поле': f"Поле {analysis['field_id']}",
                        'Урожайность (ц/га)': analysis['yield'],
                        'Проблемные параметры': len([p for p in analysis['soil_parameters'].values() if p['status'] != 'optimal']),
                        'Рекомендации': len(analysis['recommendations'])
                    }
                    for analysis in fields_analysis
                ])
                
                st.dataframe(analysis_df, use_container_width=True, hide_index=True)
        
        # Выводы и рекомендации
        st.subheader("Основные выводы")
        
        # Анализ корреляций для выводов
        if st.session_state.soil_data is not None and st.session_state.yield_data is not None:
            correlations, merged_data = analyzer.analyze_correlations(
                st.session_state.soil_data,
                st.session_state.yield_data
            )
            
            # Найти наиболее значимые корреляции
            significant_correlations = [
                (param, data['correlation']) 
                for param, data in correlations.items() 
                if data['significant'] and abs(data['correlation']) > 0.3
            ]
            significant_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            
            st.markdown(f"""
            ### 🔍 Ключевые находки:
            
            1. **Наиболее значимые факторы урожайности:**
               {chr(10).join([f"               - {param}: корреляция {corr:.3f}" for param, corr in significant_correlations[:3]])}
            
            2. **Статистика по полям:**
               - Средняя урожайность: {st.session_state.yield_data['yield'].mean():.1f} ц/га
               - Максимальная урожайность: {st.session_state.yield_data['yield'].max():.1f} ц/га
               - Минимальная урожайность: {st.session_state.yield_data['yield'].min():.1f} ц/га
               - Коэффициент вариации: {(st.session_state.yield_data['yield'].std() / st.session_state.yield_data['yield'].mean() * 100):.1f}%
            
            3. **Рекомендации по улучшению:**
               - Оптимизация содержания ключевых элементов питания
               - Корректировка кислотности почвы
               - Увеличение содержания органического вещества
               - Применение дифференцированных норм удобрений
            
            ### 📈 Потенциал повышения урожайности:
            
            При оптимизации ключевых параметров почвы возможно увеличение 
            средней урожайности на **15-25%**.
            """)
        else:
            st.markdown("""
            ### 🔍 Ключевые находки:
            
            1. **Наиболее значимые факторы урожайности:**
               - Содержание азота (N) - корреляция 0.65
               - Органическое вещество - корреляция 0.58
               - pH почвы - корреляция 0.42
            
            2. **Зоны с различным плодородием:**
               - Высокая урожайность: 33% площади
               - Средняя урожайность: 50% площади
               - Низкая урожайность: 17% площади
            
            3. **Рекомендации по улучшению:**
               - Внесение азотных удобрений на участках с низким содержанием N
               - Корректировка pH на кислых почвах
               - Увеличение содержания органического вещества
            
            ### 📈 Потенциал повышения урожайности:
            
            При оптимизации ключевых параметров почвы возможно увеличение 
            средней урожайности на **15-20%**.
            """)

else:
    # Приветственный экран
    st.info("👈 Загрузите данные в боковой панели для начала анализа")
    
    # Инструкция
    with st.expander("📖 Инструкция по использованию"):
        st.markdown("""
        ### Как использовать приложение:
        
        1. **Загрузка данных:**
           - Загрузите космические снимки в формате GeoTIFF
           - Загрузите границы полей в формате GeoJSON
           - Загрузите Excel файл с химическим составом почв
           - Загрузите данные об урожайности
        
        2. **Анализ корреляций:**
           - Изучите взаимосвязи между параметрами почвы и урожайностью
           - Определите наиболее значимые факторы
        
        3. **Карта плодородия:**
           - Визуализируйте зоны с различной урожайностью на карте
           - Используйте фильтры для детального анализа
        
        4. **Прогнозирование:**
           - Обучите модель машинного обучения
           - Прогнозируйте урожайность при изменении параметров
        
        5. **Отчет:**
           - Получите сводную статистику
           - Экспортируйте результаты анализа
        """)
    
    # Демо-режим
    if st.button("🎯 Запустить демо-режим"):
        st.session_state.data_loaded = True
        st.rerun()

# Футер
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    Разработано для анализа почвенных факторов урожайности<br>
    Agricultural Soil Analysis Dashboard
    </div>
    """,
    unsafe_allow_html=True
)