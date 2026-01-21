"""Step 3: Build baseline matcher.

Inputs (in priority order):
1) df_ideal_address (created in Step 2)  -> preferred
2) df_all_districts (created in Step 1)
3) reference_osm.csv cache

Output:
- matcher (AddressMatcher)
- df_ref (reference corpus used for fitting)
"""

import os
import sys
import pandas as pd

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
        "Then: %run code/2_make_noise.py (creates df
