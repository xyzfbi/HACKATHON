"""
Простой скрипт для создания примеров данных без внешних зависимостей
"""

import json
import random

def create_soil_csv():
    """Создание CSV файла с данными о почве"""
    random.seed(42)
    
    # Заголовки
    headers = [
        "field_id", "pH", "Organic_Matter", "N", "P", "K", 
        "Ca", "Mg", "S", "Fe", "Mn", "Zn", "Cu", "B"
    ]
    
    # Создание данных
    data_lines = [",".join(headers)]
    
    for i in range(1, 101):  # 100 образцов
        row = [
            str(i),  # field_id
            f"{random.uniform(5.5, 7.5):.2f}",  # pH
            f"{random.uniform(2.0, 5.0):.2f}",  # Organic_Matter
            f"{random.uniform(20, 50):.1f}",  # N
            f"{random.uniform(15, 40):.1f}",  # P
            f"{random.uniform(100, 200):.1f}",  # K
            f"{random.uniform(800, 1200):.1f}",  # Ca
            f"{random.uniform(100, 200):.1f}",  # Mg
            f"{random.uniform(10, 20):.1f}",  # S
            f"{random.uniform(30, 70):.1f}",  # Fe
            f"{random.uniform(20, 40):.1f}",  # Mn
            f"{random.uniform(3, 8):.2f}",  # Zn
            f"{random.uniform(2, 5):.2f}",  # Cu
            f"{random.uniform(0.5, 2.0):.2f}"  # B
        ]
        data_lines.append(",".join(row))
    
    # Сохранение в файл
    with open('example_soil_data.csv', 'w', encoding='utf-8') as f:
        f.write("\n".join(data_lines))
    
    print("✅ Создан файл: example_soil_data.csv")

def create_yield_csv():
    """Создание CSV файла с данными об урожайности"""
    random.seed(42)
    
    # Заголовки
    headers = ["field_id", "yield", "year", "crop"]
    
    # Создание данных
    data_lines = [",".join(headers)]
    
    for i in range(1, 101):  # 100 записей
        # Генерация урожайности с некоторой логикой
        base_yield = random.uniform(25, 50)
        row = [
            str(i),  # field_id
            f"{base_yield:.1f}",  # yield
            "2024",  # year
            "wheat"  # crop
        ]
        data_lines.append(",".join(row))
    
    # Сохранение в файл
    with open('example_yield_data.csv', 'w', encoding='utf-8') as f:
        f.write("\n".join(data_lines))
    
    print("✅ Создан файл: example_yield_data.csv")

def create_geojson():
    """Создание GeoJSON файла с границами полей"""
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "field_id": 1,
                    "name": "Поле 1 (Московская область)",
                    "area_ha": 70,
                    "region": "Московская область"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [37.6173, 55.7558],
                        [37.6273, 55.7558],
                        [37.6273, 55.7658],
                        [37.6173, 55.7658],
                        [37.6173, 55.7558]
                    ]]
                }
            },
            {
                "type": "Feature",
                "properties": {
                    "field_id": 2,
                    "name": "Поле 2 (Башкортостан)",
                    "area_ha": 70,
                    "region": "Башкортостан"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [55.9721, 54.7388],
                        [55.9821, 54.7388],
                        [55.9821, 54.7488],
                        [55.9721, 54.7488],
                        [55.9721, 54.7388]
                    ]]
                }
            },
            {
                "type": "Feature",
                "properties": {
                    "field_id": 3,
                    "name": "Поле 3 (Саратовская область)",
                    "area_ha": 70,
                    "region": "Саратовская область"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [46.0086, 51.5406],
                        [46.0186, 51.5406],
                        [46.0186, 51.5506],
                        [46.0086, 51.5506],
                        [46.0086, 51.5406]
                    ]]
                }
            }
        ]
    }
    
    # Сохранение в файл
    with open('example_fields.geojson', 'w', encoding='utf-8') as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)
    
    print("✅ Создан файл: example_fields.geojson")

def main():
    """Основная функция"""
    print("Создание примеров данных для тестирования приложения...")
    print()
    
    create_soil_csv()
    create_yield_csv()
    create_geojson()
    
    print()
    print("📋 Сводка созданных файлов:")
    print("- example_soil_data.csv - химический состав почв (100 образцов)")
    print("- example_yield_data.csv - данные урожайности (100 записей)")
    print("- example_fields.geojson - границы полей (3 поля)")
    print()
    print("💡 Используйте эти файлы для тестирования приложения!")
    print("📁 Файлы готовы для загрузки в приложение через интерфейс")

if __name__ == "__main__":
    main()
