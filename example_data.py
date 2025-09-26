"""
Скрипт для создания примеров данных для тестирования приложения
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime

def create_soil_data():
    """Создание примера данных о химическом составе почв"""
    np.random.seed(42)
    n_samples = 100
    
    # Генерация данных
    soil_data = pd.DataFrame({
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
    
    # Ограничение значений в разумных пределах
    soil_data['pH'] = np.clip(soil_data['pH'], 4.0, 9.0)
    soil_data['Organic_Matter'] = np.clip(soil_data['Organic_Matter'], 0.5, 8.0)
    soil_data['N'] = np.clip(soil_data['N'], 5, 80)
    soil_data['P'] = np.clip(soil_data['P'], 5, 60)
    soil_data['K'] = np.clip(soil_data['K'], 50, 300)
    
    return soil_data

def create_yield_data():
    """Создание примера данных об урожайности"""
    np.random.seed(42)
    n_samples = 100
    
    # Генерация урожайности с зависимостью от параметров почвы
    soil_data = create_soil_data()
    
    yield_base = (
        soil_data['N'] * 0.5 +
        soil_data['P'] * 0.3 +
        soil_data['K'] * 0.1 +
        soil_data['Organic_Matter'] * 5 +
        np.random.normal(0, 5, n_samples)
    )
    
    yield_data = pd.DataFrame({
        'field_id': range(1, n_samples + 1),
        'yield': yield_base + 20,
        'year': 2024,
        'crop': 'wheat'
    })
    
    # Ограничение урожайности в разумных пределах
    yield_data['yield'] = np.clip(yield_data['yield'], 10, 80)
    
    return yield_data

def create_geojson_data():
    """Создание примера GeoJSON данных с границами полей"""
    # Координаты полей в разных регионах России
    fields_coords = [
        {
            "name": "Поле 1 (Московская область)",
            "coordinates": [
                [55.7558, 37.6173],
                [55.7658, 37.6173],
                [55.7658, 37.6273],
                [55.7558, 37.6273],
                [55.7558, 37.6173]
            ]
        },
        {
            "name": "Поле 2 (Башкортостан)",
            "coordinates": [
                [54.7388, 55.9721],
                [54.7488, 55.9721],
                [54.7488, 55.9821],
                [54.7388, 55.9821],
                [54.7388, 55.9721]
            ]
        },
        {
            "name": "Поле 3 (Саратовская область)",
            "coordinates": [
                [51.5406, 46.0086],
                [51.5506, 46.0086],
                [51.5506, 46.0186],
                [51.5406, 46.0186],
                [51.5406, 46.0086]
            ]
        }
    ]
    
    # Создание GeoJSON структуры
    features = []
    for i, field in enumerate(fields_coords, 1):
        feature = {
            "type": "Feature",
            "properties": {
                "field_id": i,
                "name": field["name"],
                "area_ha": 70,
                "region": field["name"].split("(")[1].split(")")[0]
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [field["coordinates"]]
            }
        }
        features.append(feature)
    
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    return geojson

def save_example_data():
    """Сохранение примеров данных в файлы"""
    print("Создание примеров данных...")
    
    # Создание и сохранение данных о почве
    soil_data = create_soil_data()
    soil_data.to_excel('example_soil_data.xlsx', index=False)
    print("✅ Создан файл: example_soil_data.xlsx")
    
    # Создание и сохранение данных об урожайности
    yield_data = create_yield_data()
    yield_data.to_excel('example_yield_data.xlsx', index=False)
    print("✅ Создан файл: example_yield_data.xlsx")
    
    # Создание и сохранение GeoJSON данных
    geojson_data = create_geojson_data()
    with open('example_fields.geojson', 'w', encoding='utf-8') as f:
        json.dump(geojson_data, f, ensure_ascii=False, indent=2)
    print("✅ Создан файл: example_fields.geojson")
    
    print("\n📋 Сводка созданных файлов:")
    print("- example_soil_data.xlsx - химический состав почв (100 образцов)")
    print("- example_yield_data.xlsx - данные урожайности (100 записей)")
    print("- example_fields.geojson - границы полей (3 поля)")
    print("\n💡 Используйте эти файлы для тестирования приложения!")

if __name__ == "__main__":
    save_example_data()
