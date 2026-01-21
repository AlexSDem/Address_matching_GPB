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

import os
import json
import pandas as pd
import nlpaug.augmenter.char as nac

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
