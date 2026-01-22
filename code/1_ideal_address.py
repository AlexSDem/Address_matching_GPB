"""
Steps:
1) Build reference corpus from OSM (OSMnx) OR load it from cache (reference_osm.csv)
2) Create noisy queries (keyboard_noise, random_insert)
3) Fit baseline matcher (TF-IDF char n-grams + NN + RapidFuzz rerank)
4) Evaluate (Accuracy@1 + threshold table) and save results CSV

Outputs in memory:
- df_all_districts
- df_ideal_address
- matcher
- df_eval
Also saves:
- reference_osm.csv (cache)
- results_keyboard_noise.csv
"""

import os
import sys
import json
import pandas as pd
import numpy as np

# --- project root & imports from src ---
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.matcher import AddressMatcher
from src.address_normalize import normalize_ru_address

# --- optional heavy deps (only needed for Step 1 & 2) ---
import osmnx as ox
import nlpaug.augmenter.char as nac

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1200)

ox.settings.log_console = True
ox.settings.use_cache = True

CACHE_PATH = os.path.join(_ROOT, "reference_osm.csv")
RESULTS_PATH = os.path.join(_ROOT, "results_keyboard_noise.csv")
KB_JSON_PATH = os.path.join(_ROOT, "ru_keyboard.json")

# =========================================================
# Step 1: Reference corpus (OSM or cache)
# =========================================================
if os.path.exists(CACHE_PATH):
    df_all_districts = pd.read_csv(CACHE_PATH)
    print(f"✅ Loaded cached reference: {CACHE_PATH}")
    print("Shape:", df_all_districts.shape)
else:
    # You can reduce districts for a faster demo:
    city_and_distr = pd.DataFrame(
        columns=["address"],
        data=[
            "Северное Тушино, Москва, Россия",
            "Южное Тушино, Москва, Россия",
            "Центральный район, Санкт-Петербург, Россия"
            # "Октябрьский район, Новосибирск, Россия",
            # "Верх-Исетский район, Екатеринбург, Россия",
            # "Вахитовский район, Казань, Россия",
            # "Свердловский район, Красноярск, Россия",
            # "Нижегородский район, Нижний Новгород, Россия",
            # "Центральный район, Челябинск, Россия",
            # "Кировский район, Уфа, Россия",
            # "Самарский район, Самара, Россия",
            # "Ленинский район, Ростов-на-Дону, Россия",
            # "Западный округ, Краснодар, Россия",
            # "Октябрьский район, Омск, Россия",
            # "Коминтерновский район, Воронеж, Россия"
        ],
    )

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

    # city mapping
    city_dict = {}
    for place in city_and_distr["address"]:
        parts = place.split(", ")
        if len(parts) >= 3:
            city_dict[place] = ", ".join(parts[1:-1])
        else:
            city_dict[place] = ""

    all_addresses = []

    for place_name in city_and_distr["address"]:
        try:
            print(f"\n{'='*60}\nОбрабатываем: {place_name}\n{'='*60}")
            current_city = city_dict.get(place_name, "")
            print(f"  Город для подстановки: '{current_city}'")

            gdf = ox.features_from_place(place_name, tags)
            print(f"  Всего зданий в OSM: {len(gdf)}")

            existing_columns = [col for col in potential_columns if col in gdf.columns]
            address_df = pd.DataFrame(gdf[existing_columns])

            # centroids (ok for prototype)
            address_df["lat"] = gdf.geometry.centroid.y
            address_df["lon"] = gdf.geometry.centroid.x
            address_df["district"] = place_name

            address_df["united_addr"] = (
                address_df["addr:street"].fillna("") + ", " +
                address_df["addr:housenumber"].fillna("")
            ).str.strip(", ") + f", {current_city}"

            clean_addresses = address_df.dropna(subset=["addr:street", "addr:housenumber"])
            all_addresses.append(clean_addresses)

            print(f"  ✅ Чистых адресов: {len(clean_addresses)}")
            if len(clean_addresses) > 0:
                print(clean_addresses[["united_addr", "addr:street", "addr:housenumber", "district"]].head(3))

        except Exception as e:
            print(f"  ❌ Ошибка для {place_name}: {e}")

    if all_addresses:
        df_all_districts = pd.concat(all_addresses, ignore_index=True)
        print(f"\n✅ Итоговый датасет: {len(df_all_districts):,} адресов")
        print(f"🏘️ Районы: {df_all_districts['district'].nunique()}")

        df_all_districts.to_csv(CACHE_PATH, index=False)
        print(f"💾 Saved cache: {CACHE_PATH}")
    else:
        print("❌ Не удалось собрать данные ни из одного района!")
        df_all_districts = pd.DataFrame(columns=["united_addr", "district", "lat", "lon"])

# Safety check
if "united_addr" not in df_all_districts.columns or len(df_all_districts) == 0:
    raise RuntimeError(
        "Reference corpus is empty or missing 'united_addr'.\n"
        "If Overpass failed, rerun the cell/file later, or ensure reference_osm.csv exists."
    )

# =========================================================
# Step 2: Noise generation
# =========================================================
n = min(100, len(df_all_districts))
df_ideal_address = pd.DataFrame(
    df_all_districts["united_addr"].sample(n=n, random_state=42).reset_index(drop=True)
)
df_ideal_address.columns = ["united_addr"]

ru_keyboard_map = {
    'й': ['ц', 'ф', '1', '2'], 'ц': ['й', 'у', 'ф', 'ы', '2', '3'], 'у': ['ц', 'к', 'ы', 'в', '3', '4'],
    'к': ['у', 'е', 'в', 'а', '4', '5'], 'е': ['к', 'н', 'а', 'п', '5', '6'], 'н': ['е', 'г', 'п', 'р', '6', '7'],
    'г': ['н', 'ш', 'р', 'о', '7', '8'], 'ш': ['г', 'щ', 'о', 'л', '8', '9'], 'щ': ['ш', 'з', 'л', 'д', '9', '0'],
    'з': ['щ', 'х', 'д', 'ж', '0', '-'], 'х': ['з', 'ъ', 'ж', 'э', '-', '='], 'ъ': ['х', 'э', '='],
    'ф': ['й', 'ц', 'ы', 'я'], 'ы': ['ц', 'у', 'ф', 'в', 'я', 'ч'], 'в': ['у', 'к', 'ы', 'а', 'ч', 'с'],
    'а': ['к', 'е', 'в', 'п', 'с', 'м'], 'п': ['е', 'н', 'а', 'р', 'м', 'и'], 'р': ['н', 'г', 'п', 'о', 'и', 'т'],
    'о': ['г', 'ш', 'р', 'л', 'т', 'ь'], 'л': ['ш', 'щ', 'о', 'д', 'ь', 'б'], 'д': ['щ', 'з', 'л', 'ж', 'б', 'ю'],
    'ж': ['з', 'х', 'д', 'э', 'ю', '.'], 'э': ['х', 'ъ', 'ж', '.'],
    'я': ['ф', 'ы', 'ч'], 'ч': ['ы', 'в', 'я', 'с'], 'с': ['в', 'а', 'ч', 'м'], 'м': ['а', 'п', 'с', 'и'],
    'и': ['п', 'р', 'м', 'т'], 'т': ['р', 'о', 'и', 'ь'], 'ь': ['о', 'л', 'т', 'б'], 'б': ['л', 'д', 'ь', 'ю'],
    'ю': ['д', 'ж', 'б', '.']
}

with open(KB_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(ru_keyboard_map, f, ensure_ascii=False)

aug_keyboard = nac.KeyboardAug(
    model_path=KB_JSON_PATH,
    aug_char_p=0.2,
    aug_word_p=0.1,
)

aug_random = nac.RandomCharAug(
    action="insert",
    aug_char_p=0.2,
    aug_word_p=0.1,
    spec_char="!@#%_123",
)

df_ideal_address["keyboard_noise"] = df_ideal_address["united_addr"].apply(lambda x: aug_keyboard.augment(x)[0])
df_ideal_address["random_insert"] = df_ideal_address["united_addr"].apply(lambda x: aug_random.augment(x)[0])

print("\n✅ df_ideal_address created:", df_ideal_address.shape)
print(df_ideal_address.head(3))

# =========================================================
# Step 3: Fit matcher
# =========================================================
df_ref = pd.DataFrame({"united_addr": df_all_districts["united_addr"]}).dropna()
if len(df_ref) == 0:
    raise RuntimeError("Reference corpus is empty after dropna().")

matcher = AddressMatcher(
    ngram_range=(2, 4),
    analyzer="char_wb",
    top_k=10,
    w_cosine=0.6,
    w_fuzz=0.4,
    do_normalize=True,
).fit(df_ref["united_addr"].tolist())

print("\n✅ Matcher fitted on:", len(df_ref))

# =========================================================
# Step 4: Evaluate
# =========================================================
def _threshold_report(df: pd.DataFrame, thresholds=(0.70, 0.75, 0.80, 0.85, 0.90)) -> pd.DataFrame:
    out = []
    for t in thresholds:
        auto = df[df["final_score"] >= t]
        coverage = len(auto) / len(df) if len(df) else 0.0
        precision = float(auto["is_correct"].mean()) if len(auto) else 0.0
        recall = float(((df["final_score"] >= t) & (df["is_correct"]).astype(bool)).mean()) if len(df) else 0.0
        out.append({"threshold": t, "precision": precision, "recall": recall, "coverage": coverage})
    return pd.DataFrame(out)

df_eval = pd.DataFrame(
    {
        "query": df_ideal_address["keyboard_noise"],
        "true": df_ideal_address["united_addr"],
    }
)

pred = matcher.match_batch(df_eval["query"].tolist())

# IMPORTANT: avoid overlapping column names (pred contains 'query')
df_eval = df_eval.join(pred.drop(columns=["query"]))

df_eval["true_norm"] = df_eval["true"].map(normalize_ru_address)
df_eval["best_norm"] = df_eval["best"].map(normalize_ru_address)
df_eval["is_correct"] = df_eval["true_norm"] == df_eval["best_norm"]

acc1 = float(df_eval["is_correct"].mean()) if len(df_eval) else 0.0
print(f"\nAccuracy@1 (keyboard_noise): {acc1:.3f}  (n={len(df_eval)})")

report = _threshold_report(df_eval)
print("\nThreshold report (auto-match by final_score):")
print(report.to_string(index=False, formatters={
    "threshold": "{:.2f}".format,
    "precision": "{:.3f}".format,
    "recall": "{:.3f}".format,
    "coverage": "{:.3f}".format,
}))

df_eval.to_csv(RESULTS_PATH, index=False)
print(f"\n✅ Saved: {RESULTS_PATH}")

# Quick sanity example for demo
sample_query = df_eval.loc[0, "query"]
sample_best = df_eval.loc[0, "best"]
sample_score = df_eval.loc[0, "final_score"]
print("\nDemo example:")
print(" query:", sample_query)
print(" best :", sample_best)
print(" score:", round(float(sample_score), 3))
