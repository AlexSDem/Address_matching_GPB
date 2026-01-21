"""
Step 1: Build reference corpus from OSM (OSMnx) OR load it from cache.

Output:
- df_all_districts (DataFrame) with at least column 'united_addr'
- also saves cache file: reference_osm.csv in repo root
"""

import os
import pandas as pd
import numpy as np
import osmnx as ox
import json
import nlpaug.augmenter.char as nac
import sys

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


"""
Step 2: Create noisy queries from reference corpus.

Input:
- df_all_districts from Step 1 (in memory),
  OR loads cache reference_osm.csv if df_all_districts is missing.

Output:
- df_ideal_address with columns:
  - united_addr (ground truth)
  - keyboard_noise
  - random_insert
"""

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE_PATH = os.path.join(_ROOT, "reference_osm.csv")

# 1) Ensure df_all_districts exists
if "df_all_districts" not in globals() or df_all_districts is None or len(df_all_districts) == 0:
    if os.path.exists(CACHE_PATH):
        df_all_districts = pd.read_csv(CACHE_PATH)
        print(f"✅ Loaded cached reference: {CACHE_PATH} shape={df_all_districts.shape}")
    else:
        raise RuntimeError(
            "df_all_districts is not defined and cache file reference_osm.csv not found.\n"
            "Run: %run code/1_ideal_address.py (it will create df_all_districts and save cache)."
        )

if "united_addr" not in df_all_districts.columns:
    raise RuntimeError("df_all_districts has no column 'united_addr'. Check code/1_ideal_address.py output.")

# 2) Sample
n = min(100, len(df_all_districts))
df_ideal_address = pd.DataFrame(
    df_all_districts["united_addr"].sample(n=n, random_state=42).reset_index(drop=True)
)
df_ideal_address.columns = ["united_addr"]

# 3) Keyboard map (RU)
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

kb_path = os.path.join(_ROOT, "ru_keyboard.json")
with open(kb_path, "w", encoding="utf-8") as f:
    json.dump(ru_keyboard_map, f, ensure_ascii=False)

aug_keyboard = nac.KeyboardAug(
    model_path=kb_path,
    aug_char_p=0.2,
    aug_word_p=0.1,  # 10% слов, а не 100%
)

aug_random = nac.RandomCharAug(
    action="insert",
    aug_char_p=0.2,
    aug_word_p=0.1,
    spec_char="!@#%_123",
)

df_ideal_address["keyboard_noise"] = df_ideal_address["united_addr"].apply(lambda x: aug_keyboard.augment(x)[0])
df_ideal_address["random_insert"] = df_ideal_address["united_addr"].apply(lambda x: aug_random.augment(x)[0])

print("✅ df_ideal_address created:", df_ideal_address.shape)
print(df_ideal_address.head(3))


"""Step 3: Build baseline matcher.

Inputs (in priority order):
1) df_ideal_address (created in Step 2)  -> preferred
2) df_all_districts (created in Step 1)
3) reference_osm.csv cache

Output:
- matcher (AddressMatcher)
- df_ref (reference corpus used for fitting)
"""

# Allow running from /code
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.matcher import AddressMatcher

CACHE_PATH = os.path.join(_ROOT, "reference_osm.csv")

# 1) Ensure we have some reference addresses to fit on
ref_series = None

if "df_ideal_address" in globals() and df_ideal_address is not None and len(df_ideal_address) > 0:
    if "united_addr" not in df_ideal_address.columns:
        raise RuntimeError("df_ideal_address has no column 'united_addr'.")
    ref_series = df_ideal_address["united_addr"]
    print("✅ Using df_ideal_address as reference:", len(ref_series))

elif "df_all_districts" in globals() and df_all_districts is not None and len(df_all_districts) > 0:
    if "united_addr" not in df_all_districts.columns:
        raise RuntimeError("df_all_districts has no column 'united_addr'.")
    ref_series = df_all_districts["united_addr"]
    print("✅ Using df_all_districts as reference:", len(ref_series))

elif os.path.exists(CACHE_PATH):
    df_all_districts = pd.read_csv(CACHE_PATH)
    if "united_addr" not in df_all_districts.columns:
        raise RuntimeError("reference_osm.csv has no column 'united_addr'.")
    ref_series = df_all_districts["united_addr"]
    print(f"✅ Loaded cache reference_osm.csv as reference: {len(ref_series)}")

else:
    raise RuntimeError(
        "No reference data found.\n"
        "Run: %run code/1_ideal_address.py (creates df_all_districts + saves reference_osm.csv)\n"
        "Then: %run code/2_make_noise.py (creates df_ideal_address)\n"
    )

# 2) Build reference DataFrame and fit matcher
df_ref = pd.DataFrame({"united_addr": ref_series}).dropna()
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

print("✅ Matcher fitted on:", len(df_ref))

# Quick sanity-check
bad_address = "ул труд, челяба"
res = matcher.match_one(bad_address)
print(f"Sanity check:\n  query={bad_address}\n  best={res.best}\n  final={res.final_score:.3f}")


"""Evaluation step (baseline/prototype).

Expected inputs from previous steps (same Python session):
- df_ideal_address: DataFrame with columns:
    - united_addr (ground truth reference)
    - keyboard_noise (noisy query)  [created in 2_make_noise.py]
  (Optionally: random_insert)
- matcher: fitted AddressMatcher (created in 3_model.py)

Outputs:
- df_eval: dataframe with predictions + scores
- prints: Accuracy@1 + a simple threshold table

Note: because we synthesize noise from united_addr, we know the ground truth.
"""

# Allow running this file from within the /code folder or from notebooks
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

from src.address_normalize import normalize_ru_address


def _threshold_report(df: pd.DataFrame, thresholds=(0.70, 0.75, 0.80, 0.85, 0.90)) -> pd.DataFrame:
    """Precision/Recall/Coverage for an 'auto-match' threshold.

    - auto-match if final_score >= T
    - precision: among auto-matches, how many are correct
    - recall: among all queries, how many correct auto-matches we got
    - coverage: share of queries auto-matched
    """
    out = []
    for t in thresholds:
        auto = df[df["final_score"] >= t]
        coverage = len(auto) / len(df) if len(df) else 0.0
        if len(auto) == 0:
            precision = 0.0
        else:
            precision = float((auto["is_correct"]).mean())
        recall = float(((df["final_score"] >= t) & (df["is_correct"]).astype(bool)).mean())
        out.append({"threshold": t, "precision": precision, "recall": recall, "coverage": coverage})
    return pd.DataFrame(out)


# 1) Build evaluation dataset
df_eval = pd.DataFrame(
    {
        "query": df_ideal_address["keyboard_noise"],
        "true": df_ideal_address["united_addr"],
    }
)

# 2) Predict
pred = matcher.match_batch(df_eval["query"].tolist())
df_eval = df_eval.join(pred)

# 3) Compare (with normalization to be robust to formatting)
df_eval["true_norm"] = df_eval["true"].map(normalize_ru_address)
df_eval["best_norm"] = df_eval["best"].map(normalize_ru_address)
df_eval["is_correct"] = df_eval["true_norm"] == df_eval["best_norm"]

acc1 = float(df_eval["is_correct"].mean()) if len(df_eval) else 0.0

print(f"Accuracy@1 (keyboard_noise): {acc1:.3f}  (n={len(df_eval)})")

# 4) Show a simple threshold trade-off table
report = _threshold_report(df_eval)
print("\nThreshold report (auto-match by final_score):")
print(report.to_string(index=False, formatters={
    "threshold": "{:.2f}".format,
    "precision": "{:.3f}".format,
    "recall": "{:.3f}".format,
    "coverage": "{:.3f}".format,
}))

# 5) Save results for demo
out_path = os.path.join(_ROOT, "results_keyboard_noise.csv")
df_eval.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
