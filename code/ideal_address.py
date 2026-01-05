# Собираем города и районы - вручную, пока не найден более оптимальный способ
city_and_distr = pd.DataFrame(columns=['address'],
                              data=['Северное Тушино, Москва, Россия',
                                    'Южное Тушино, Москва, Россия',
                                    'Центральный район, Санкт-Петербург, Россия',
                                    'Октябрьский район, Новосибирск, Россия',
                                    'Верх-Исетский район, Екатеринбург, Россия',
                                    'Вахитовский район, Казань, Россия',
                                    'Свердловский район, Красноярск, Россия',
                                    'Нижегородский район, Нижний Новгород, Россия',
                                    'Центральный район, Челябинск, Россия',
                                    'Кировский район, Уфа, Россия',
                                    'Самарский район, Самара, Россия',
                                    'Ленинский район, Ростов-на-Дону, Россия',
                                    'Западный округ, Краснодар, Россия',
                                    'Октябрьский район, Омск, Россия',
                                    'Коминтерновский район, Воронеж, Россия'])

# Здесь по нескольким адресам
# 1.1 UPGRADE: Полный код для нескольких районов OSMnx

import osmnx as ox
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

ox.settings.log_console = True
ox.settings.use_cache = True

# 1. Настройки
tags = {"building": True}

potential_columns = [
    'addr:city',
    'addr:street',
    'addr:housenumber',
    'addr:postcode',
    'addr:flats',
    'addr:district',
    'addr:suburb',
    'name',
    'building'
]

# districts = [
#     "Южное Тушино, Москва, Россия",
#     "Северное Тушино, Москва, Россия"
# ]

all_addresses = []

# Словарь городов из city_and_distr['address']
city_dict = {}
for place in city_and_distr['address']:
    # Извлекаем город: всё после района, до ", Россия"
    parts = place.split(', ')
    if len(parts) >= 3:
        district = parts[0]
        city = ', '.join(parts[1:-1])  # "Северное Тушино, Москва" → "Москва"
        city_dict[place] = city
    else:
        city_dict[place] = ''  # fallback

# 2. Собираем данные по районам
for place_name in city_and_distr['address']:
    try:
        print(f"\n{'='*60}")
        print(f"Обрабатываем: {place_name}")
        print(f"{'='*60}")
        current_city = city_dict.get(place_name, '')
        print(f"  Город для подстановки: '{current_city}'")

        gdf = ox.features_from_place(place_name, tags)
        print(f"  Всего зданий в OSM: {len(gdf)}")

        # Фильтруем доступные колонки
        existing_columns = [col for col in potential_columns if col in gdf.columns]
        print(f"  Доступные колонки: {len(existing_columns)}")

        # Создаём DataFrame
        address_df = pd.DataFrame(gdf[existing_columns])

        # Добавляем координаты центра здания
        address_df['lat'] = gdf.geometry.centroid.y
        address_df['lon'] = gdf.geometry.centroid.x

        # Метка района
        address_df['district'] = place_name

        # Формируем "объединённый адрес" для TF-IDF
        address_df['united_addr'] = (
            address_df['addr:street'].fillna('') + ', ' +
            address_df['addr:housenumber'].fillna('') + ', '
            # address_df.get('addr:city', pd.Series(['']*len(address_df))).fillna('')
        ).str.strip(', ') + f", {current_city}"

        # Очищаем: только адреса с улицей И номером дома
        clean_addresses = address_df.dropna(subset=['addr:street', 'addr:housenumber'])

        all_addresses.append(clean_addresses)
        print(f"  ✅ Найдено чистых адресов: {len(clean_addresses)}")

        # Показываем примеры
        if len(clean_addresses) > 0:
            print("  Примеры адресов:")
            print(clean_addresses[['united_addr', 'addr:street', 'addr:housenumber', 'district']].head(3))

    except Exception as e:
        print(f"  ❌ Ошибка для {place_name}: {e}")

# 3. Объединяем все районы
if all_addresses:
    df_all_districts = pd.concat(all_addresses, ignore_index=True)

    print(f"\n{'='*60}")
    print(f"ИТОГОВЫЙ ДАТАСЕТ")
    print(f"{'='*60}")
    print(f"📊 Всего адресов: {len(df_all_districts):,}")
    print(f"🏘️  Районы: {df_all_districts['district'].nunique()}")
    print(f"📍  Уникальных улиц: {df_all_districts['addr:street'].nunique()}")

    print("\nРаспределение по районам:")
    print(df_all_districts['district'].value_counts())

    print("\nПервые 5 адресов:")
    display_cols = ['united_addr', 'addr:street', 'addr:housenumber', 'district', 'lat', 'lon']
    print(df_all_districts[display_cols].head())

    # Сохраняем для TF-IDF
    print(f"\n💾 Готово! df_all_districts['united_addr'] — для векторизации TF-IDF")

    # df_ideal_address = df_all_districts  # Раскомментируйте для основного кода

else:
    print("❌ Не удалось собрать данные ни из одного района!")
    df_all_districts = pd.DataFrame()
