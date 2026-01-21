"""
Step 1: Build reference corpus from OSM (OSMnx) OR load it from cache.

Output:
- df_all_districts (DataFrame) with at least column 'united_addr'
- also saves cache file: reference_osm.csv in repo root
"""

import os
import pandas as pd
import osmnx as ox

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1200)

ox.settings.log_console = True
ox.settings.use_cache = True

# --- Cache ---
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE_PATH = os.path.join(_ROOT, "reference_osm.csv")

# If cache exists, load and exit early (stable for Colab demos)
if os.path.exists(CACHE_PATH):
    df_all_districts = pd.read_csv(CACHE_PATH)
    print(f"✅ Loaded cached reference: {CACHE_PATH}")
    print("Shape:", df_all_districts.shape)
else:
    # Собираем города и районы - вручную
    city_and_distr = pd.DataFrame(
        columns=["address"],
        data=[
            "Северное Тушино, Москва, Россия",
            "Южное Тушино, Москва, Россия",
            "Центральный район, Санкт-Петербург, Россия",
            "Октябрьский район, Новосибирск, Россия",
            "Верх-Исетский район, Екатеринбург, Россия",
            "Вахитовский район, Казань, Россия",
            "Свердловский район, Красноярск, Россия",
            "Нижегородский район, Нижний Новгород, Россия",
            "Центральный район, Челябинск, Россия",
            "Кировский район, Уфа, Россия",
            "Самарский район, Самара, Россия",
            "Ленинский район, Ростов-на-Дону, Россия",
            "Западный округ, Краснодар, Россия",
            "Октябрьский район, Омск, Россия",
            "Коминтерновский район, Воронеж, Россия",
        ],
    )

    # 1. Настройки
    tags = {"building": True}
    potential_columns = [
        "addr:city",
        "addr:street",
        "addr:housenumber",
        "addr:postcode",
        "addr:flats",
        "addr:district",
        "addr:suburb",
        "name",
        "building",
    ]

    all_addresses = []

    # Словарь городов
    city_dict = {}
    for place in city_and_distr["address"]:
        parts = place.split(", ")
        if len(parts) >= 3:
            city = ", ".join(parts[1:-1])  # всё между районом и "Россия"
            city_dict[place] = city
        else:
            city_dict[place] = ""

    # 2. Собираем данные по районам
    for place_name in city_and_distr["address"]:
        try:
            print(f"\n{'='*60}\nОбрабатываем: {place_name}\n{'='*60}")
            current_city = city_dict.get(place_name, "")
            print(f"  Город для подстановки: '{current_city}'")

            gdf = ox.features_from_place(place_name, tags)
            print(f"  Всего зданий в OSM: {len(gdf)}")

            existing_columns = [col for col in potential_columns if col in gdf.columns]
            print(f"  Доступные колонки: {len(existing_columns)}")

            address_df = pd.DataFrame(gdf[existing_columns])

            # Добавляем координаты центра здания (может быть медленно, но ок для прототипа)
            address_df["lat"] = gdf.geometry.centroid.y
            address_df["lon"] = gdf.geometry.centroid.x

            address_df["district"] = place_name

            # Формируем united_addr
            address_df["united_addr"] = (
                address_df["addr:street"].fillna("") + ", " +
                address_df["addr:housenumber"].fillna("") 
            ).str.strip(", ") + f", {current_city}"

            # только строки с улицей и домом
            clean_addresses = address_df.dropna(subset=["addr:street", "addr:housenumber"])

            all_addresses.append(clean_addresses)
            print(f"  ✅ Найдено чистых адресов: {len(clean_addresses)}")

            if len(clean_addresses) > 0:
                print(clean_addresses[["united_addr", "addr:street", "addr:housenumber", "district"]].head(3))

        except Exception as e:
            print(f"  ❌ Ошибка для {place_name}: {e}")

    # 3. Объединяем
    if all_addresses:
        df_all_districts = pd.concat(all_addresses, ignore_index=True)
        print(f"\n✅ Итоговый датасет: {len(df_all_districts):,} адресов")
        print(f"🏘️ Районы: {df_all_districts['district'].nunique()}")

        # Save cache
        df_all_districts.to_csv(CACHE_PATH, index=False)
        print(f"💾 Saved cache: {CACHE_PATH}")

    else:
        print("❌ Не удалось собрать данные ни из одного района!")
        df_all_districts = pd.DataFrame(columns=["united_addr", "district", "lat", "lon"])
