import os
import re
import uuid
import math
from datetime import datetime, date, time
from typing import Union, List, Dict, Any

import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

# KONFIGURASI

OPERATOR_DIHAPUS = ["devi", "dina", "owner"]

TIPE_PENJUALAN_DIHAPUS = ["manual", "agen", "member", "reseller"]

KATA_NON_PRODUK = [
    "kardus",
    "lakban",
    "brosur",
    "clutch",
    "pouch",
    "boxi",
    "paper bag",
    "paperbag",
    "bubble wrap",
    "tas",
    "plastik",
    "tenteng",
    "cermin",
    "ongkir",
]

COLUMN_ALIASES = {
    "no transaksi": "no_transaksi",
    "no_transaksi": "no_transaksi",
    "notransaksi": "no_transaksi",
    "no. transaksi": "no_transaksi",

    "tanggal": "tanggal",
    "waktu": "waktu",
    "operator": "operator",

    "tipe penjualan": "tipe_penjualan",
    "tipe_penjualan": "tipe_penjualan",
    "type penjualan": "tipe_penjualan",
    "jenis penjualan": "tipe_penjualan",

    "detail produk": "detail_produk",
    "detail_produk": "detail_produk",
    "detail product": "detail_produk",
    "nama produk": "detail_produk",
    "produk": "detail_produk",

    "penjualan bersih": "penjualan_bersih",
    "penjualan_bersih": "penjualan_bersih",
    "penjualan": "penjualan_bersih",
    "total penjualan": "penjualan_bersih",
    "harga jual": "penjualan_bersih",
    "subtotal": "penjualan_bersih",
}

REQUIRED_COLUMNS = [
    "no_transaksi",
    "detail_produk",
    "penjualan_bersih",
]

# Parameter default disamakan dengan model.py
DEFAULT_MIN_SUPPORT = 0.02
DEFAULT_MIN_CONFIDENCE = 0.6
DEFAULT_MIN_LIFT = 1.0

# HELPER
# membaca file excel, membersihkan data, menyiapkan dataset untuk FP-Growth, menjalankan FP-Growth, dan membangun association rules

def _ensure_list(file_paths: Union[str, List[str]]) -> List[str]:
    if isinstance(file_paths, list):
        return file_paths
    return [file_paths]


def _is_missing_value(value) -> bool:
    if value is None:
        return True

    if isinstance(value, (list, tuple, set, frozenset, dict)):
        return False

    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _safe_text(value):
    if _is_missing_value(value):
        return None

    text = str(value).strip()

    if text == "" or text.lower() in ["nan", "none", "null", "<na>", "nat"]:
        return None

    return text


def _safe_join(items, separator=", "):
    if not isinstance(items, (list, tuple, set, frozenset)):
        text = _safe_text(items)
        return text if text is not None else ""

    clean_items = []

    for item in items:
        text = _safe_text(item)
        if text is not None:
            clean_items.append(text)

    return separator.join(sorted(clean_items))


def _merge_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kalau setelah rename ada nama kolom dobel,
    gabungkan dengan mengambil nilai pertama yang tidak kosong.
    """
    result = pd.DataFrame(index=df.index)

    for idx, col in enumerate(list(df.columns)):
        series = df.iloc[:, idx]

        if col not in result.columns:
            result[col] = series
        else:
            result[col] = result[col].combine_first(series)

    return result


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    rename_map = {}

    for col in df.columns:
        key = col.strip().lower()
        if key in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[key]

    df = df.rename(columns=rename_map)
    df = _merge_duplicate_columns(df)

    return df


def _read_excel_files(file_paths: Union[str, List[str]]) -> pd.DataFrame:
    files = _ensure_list(file_paths)
    df_list = []

    for file_path in files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File tidak ditemukan: {file_path}")

        raw = pd.read_excel(file_path, header=None)

        header_row_index = None

        for i in range(min(30, len(raw))):
            row_values = raw.iloc[i].tolist()
            row_text = " | ".join(
                str(value).strip().lower()
                for value in row_values
                if _safe_text(value) is not None
            )

            has_no_transaksi = "no transaksi" in row_text
            has_detail_produk = "detail produk" in row_text or "detail product" in row_text
            has_penjualan = "penjualan" in row_text

            if has_no_transaksi and has_detail_produk and has_penjualan:
                header_row_index = i
                break

        if header_row_index is None:
            raise ValueError(
                "Header tabel tidak ditemukan. "
                "Pastikan dataset memiliki baris header yang berisi kolom: "
                "No Transaksi, Detail Produk, dan Penjualan."
            )

        df = pd.read_excel(file_path, header=header_row_index)
        df.columns = df.columns.astype(str).str.strip()

        df = df.loc[:, ~df.columns.str.contains("^Unnamed", case=False, na=False)]
        df = df.dropna(how="all")

        df["source_file"] = os.path.basename(file_path)

        df_list.append(df)

    if len(df_list) == 0:
        raise ValueError("Tidak ada file Excel yang berhasil dibaca.")

    df_all = pd.concat(df_list, ignore_index=True)
    return df_all


def _clean_number(value):
    if _is_missing_value(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip()

    if text == "" or text.lower() in ["nan", "none", "null", "<na>"]:
        return np.nan

    text = re.sub(r"[^\d,.\-]", "", text)

    if text in ["", "-", ".", ","]:
        return np.nan

    if "," in text and "." in text:
        text = text.replace(".", "")
        text = text.replace(",", ".")
    elif "," in text:
        text = text.replace(",", ".")
    elif "." in text:
        parts = text.split(".")
        if len(parts) > 1 and all(len(part) == 3 for part in parts[1:]):
            text = "".join(parts)

    try:
        return float(text)
    except ValueError:
        return np.nan


def _parse_time_value(value):
    if _is_missing_value(value):
        return None, np.nan

    if isinstance(value, time):
        return value.strftime("%H:%M"), value.hour

    if isinstance(value, datetime):
        return value.strftime("%H:%M"), value.hour

    if isinstance(value, pd.Timestamp):
        return value.strftime("%H:%M"), value.hour

    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)

        if 0 <= number < 1:
            total_minutes = int(round(number * 24 * 60))
            hour = (total_minutes // 60) % 24
            minute = total_minutes % 60
            return f"{hour:02d}:{minute:02d}", hour

        parsed = pd.to_datetime(
            number,
            origin="1899-12-30",
            unit="D",
            errors="coerce"
        )

        if pd.notna(parsed):
            return parsed.strftime("%H:%M"), parsed.hour

    parsed = pd.to_datetime(str(value), errors="coerce")

    if pd.notna(parsed):
        return parsed.strftime("%H:%M"), parsed.hour

    return None, np.nan


def _kategori_waktu(jam):
    """
    Pembagian waktu terbaru:
    - Pagi  = 00.00 sampai 11.59
    - Siang = 12.00 sampai 23.59
    """
    if _is_missing_value(jam):
        return pd.NA

    jam = int(jam)

    if 0 <= jam < 12:
        return "Pagi"

    if 12 <= jam <= 23:
        return "Siang"

    return pd.NA


def _kategori_kanal_penjualan(tipe_penjualan):
    tipe_penjualan = _safe_text(tipe_penjualan)

    if tipe_penjualan is None:
        return pd.NA

    tipe_penjualan = tipe_penjualan.lower().strip()

    if tipe_penjualan == "contactless dining":
        return "offline"
    elif tipe_penjualan in ["shopee", "online order"]:
        return "online"
    else:
        return pd.NA


def _make_token(prefix: str, value):
    text = _safe_text(value)

    if text is None:
        return pd.NA

    return f"{prefix}{text}"


def _itemset_to_text(itemset):
    return _safe_join(itemset)


def _item_token_to_display(item):
    text = _safe_text(item)

    if text is None:
        return ""

    if text.startswith("produk_"):
        return text.replace("produk_", "")

    if text.startswith("operator_"):
        return "Operator: " + text.replace("operator_", "")

    if text.startswith("waktu_"):
        return "Waktu: " + text.replace("waktu_", "")

    return text


def _itemset_to_display(itemset):
    if isinstance(itemset, (set, frozenset, list, tuple)):
        display_items = []

        for item in itemset:
            display_item = _item_token_to_display(item)

            if display_item != "":
                display_items.append(display_item)

        return _safe_join(display_items)

    return _item_token_to_display(itemset)


def _jsonable(value):
    if isinstance(value, (set, frozenset, list, tuple)):
        return [_jsonable(item) for item in value]

    if isinstance(value, (datetime, date, time, pd.Timestamp)):
        return value.isoformat()

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if _is_missing_value(value):
        return None

    return value


def _df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    records = df.to_dict(orient="records")

    clean_records = []

    for row in records:
        clean_row = {}

        for key, value in row.items():
            clean_row[key] = _jsonable(value)

        clean_records.append(clean_row)

    return clean_records

# KATEGORI RULE + DETEKSI ANOMALI
# Mengikuti model.py yang udah dibuat

def categorize_rule(row: pd.Series) -> str:
    confidence = float(row["confidence"])
    lift = float(row["lift"])

    if confidence >= 0.55 and lift >= 1.3:
        return "Strong Pattern"
    elif confidence >= 0.4 and lift > 1.0:
        return "Moderate Pattern"
    else:
        return "Weak Pattern"


def add_rule_category_and_anomaly(rules: pd.DataFrame) -> pd.DataFrame:
    """
    Menambahkan:
    - kategori_rule: Strong Pattern / Moderate Pattern / Weak Pattern
    - is_anomaly: deteksi anomali menggunakan IQR pada lift dan confidence

    Logic ini mengikuti model.py.
    """
    rules = rules.copy()

    if rules.empty:
        rules["kategori_rule"] = []
        rules["is_anomaly"] = []
        return rules

    rules["confidence"] = pd.to_numeric(rules["confidence"], errors="coerce")
    rules["lift"] = pd.to_numeric(rules["lift"], errors="coerce")

    q1_lift = rules["lift"].quantile(0.25)
    q3_lift = rules["lift"].quantile(0.75)
    iqr_lift = q3_lift - q1_lift

    q1_conf = rules["confidence"].quantile(0.25)
    q3_conf = rules["confidence"].quantile(0.75)
    iqr_conf = q3_conf - q1_conf

    lower_lift = q1_lift - 1.5 * iqr_lift
    upper_lift = q3_lift + 1.5 * iqr_lift

    lower_conf = q1_conf - 1.5 * iqr_conf
    upper_conf = q3_conf + 1.5 * iqr_conf

    rules["is_anomaly"] = (
        (rules["lift"] < lower_lift) |
        (rules["lift"] > upper_lift) |
        (rules["confidence"] < lower_conf) |
        (rules["confidence"] > upper_conf)
    )

    rules["is_anomaly"] = rules["is_anomaly"].fillna(False)
    rules["kategori_rule"] = rules.apply(categorize_rule, axis=1)

    return rules


def build_rules_export(rules_df: pd.DataFrame) -> pd.DataFrame:
    if rules_df.empty:
        return pd.DataFrame(
            columns=[
                "kanal_filter",
                "antecedents_raw",
                "consequents_raw",
                "antecedents_display",
                "consequents_display",
                "antecedent support",
                "consequent support",
                "support",
                "confidence",
                "lift",
                "leverage",
                "conviction",
                "zhangs_metric",
                "jaccard",
                "certainty",
                "kulczynski",
                "kategori_rule",
                "is_anomaly",
            ]
        )

    rules_export = rules_df.copy()

    rules_export["antecedents_raw"] = rules_export["antecedents"].apply(_itemset_to_text)
    rules_export["consequents_raw"] = rules_export["consequents"].apply(_itemset_to_text)

    rules_export["antecedents_display"] = rules_export["antecedents"].apply(_itemset_to_display)
    rules_export["consequents_display"] = rules_export["consequents"].apply(_itemset_to_display)

    selected_rule_columns = [
        "kanal_filter",
        "antecedents_raw",
        "consequents_raw",
        "antecedents_display",
        "consequents_display",
        "antecedent support",
        "consequent support",
        "support",
        "confidence",
        "lift",
        "leverage",
        "conviction",
        "zhangs_metric",
        "jaccard",
        "certainty",
        "kulczynski",
        "kategori_rule",
        "is_anomaly",
    ]

    selected_rule_columns = [
        col for col in selected_rule_columns if col in rules_export.columns
    ]

    rules_export = rules_export[selected_rule_columns]

    return rules_export

# PREPROCESSING DATA MENTAH

def preprocess_raw_dataframe(df_raw: pd.DataFrame):
    total_data_awal = int(df_raw.shape[0])

    df = _standardize_columns(df_raw)

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_columns:
        raise ValueError(
            "Kolom wajib tidak ditemukan pada dataset: "
            + ", ".join(missing_columns)
            + ". Pastikan dataset memiliki kolom No Transaksi, Detail Produk, dan Penjualan."
        )

    selected_columns = [
        "no_transaksi",
        "tanggal",
        "waktu",
        "operator",
        "tipe_penjualan",
        "detail_produk",
        "penjualan_bersih",
        "source_file",
    ]

    available_columns = [col for col in selected_columns if col in df.columns]
    df = df[available_columns].copy()

    if "tanggal" not in df.columns:
        df["tanggal"] = pd.NA

    if "waktu" not in df.columns:
        df["waktu"] = pd.NA

    if "operator" not in df.columns:
        df["operator"] = pd.NA

    if "tipe_penjualan" not in df.columns:
        df["tipe_penjualan"] = pd.NA

    if "source_file" not in df.columns:
        df["source_file"] = pd.NA

    df["no_transaksi"] = (
        df["no_transaksi"]
        .astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.upper()
    )

    df["detail_produk"] = (
        df["detail_produk"]
        .astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.lower()
    )

    df["operator"] = (
        df["operator"]
        .astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.lower()
    )

    df["tipe_penjualan"] = (
        df["tipe_penjualan"]
        .astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.lower()
    )

    df["penjualan_bersih"] = df["penjualan_bersih"].apply(_clean_number)

    df = df.dropna(subset=["no_transaksi", "detail_produk", "penjualan_bersih"])

    df = df[
        (df["no_transaksi"].astype(str).str.strip() != "")
        & (df["detail_produk"].astype(str).str.strip() != "")
        & (~df["no_transaksi"].astype(str).str.lower().isin(["nan", "none", "null", "<na>"]))
        & (~df["detail_produk"].astype(str).str.lower().isin(["nan", "none", "null", "<na>"]))
    ]

    df = df[~df["tipe_penjualan"].isin(TIPE_PENJUALAN_DIHAPUS)]

    df["kanal_penjualan"] = df["tipe_penjualan"].apply(_kategori_kanal_penjualan)
    df = df[df["kanal_penjualan"].notna()]

    df = df[~df["operator"].isin(OPERATOR_DIHAPUS)]

    df["detail_produk"] = df["detail_produk"].str.replace(
        r"\bpkt\b",
        "paket",
        regex=True
    )

    pattern_non_produk = "|".join(re.escape(kata) for kata in KATA_NON_PRODUK)

    df = df[
        ~df["detail_produk"].str.contains(
            pattern_non_produk,
            case=False,
            na=False
        )
    ]

    df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce").dt.date

    parsed_time = df["waktu"].apply(_parse_time_value)

    df["waktu"] = parsed_time.apply(lambda x: x[0])
    df["jam"] = parsed_time.apply(lambda x: x[1])
    df["kategori_waktu"] = df["jam"].apply(_kategori_waktu)
    df = df.drop(columns=["jam"])

    df = df[df["kategori_waktu"].notna()]

    df = df[df["penjualan_bersih"] > 0]

    df["operator"] = df["operator"].astype("string").str.strip().str.title()
    df["kanal_penjualan"] = df["kanal_penjualan"].astype("string").str.strip().str.lower()

    duplicate_subset = [col for col in df.columns if col != "source_file"]
    jumlah_duplikat = int(df.duplicated(subset=duplicate_subset).sum())

    df = df.drop_duplicates(subset=duplicate_subset)
    df = df.reset_index(drop=True)

    summary_preprocessing = {
        "total_data_awal": total_data_awal,
        "setelah_preprocessing": int(df.shape[0]),
        "jumlah_data_dihapus": int(total_data_awal - df.shape[0]),
        "jumlah_duplikat_dihapus": jumlah_duplikat,
        "jumlah_transaksi_unik": int(df["no_transaksi"].nunique()),
        "jumlah_produk_unik": int(df["detail_produk"].nunique()),
        "jumlah_operator_unik": int(df["operator"].nunique()),
        "jumlah_kanal_penjualan_unik": int(df["kanal_penjualan"].nunique()),
        "jumlah_transaksi_offline": int(df[df["kanal_penjualan"] == "offline"]["no_transaksi"].nunique()),
        "jumlah_transaksi_online": int(df[df["kanal_penjualan"] == "online"]["no_transaksi"].nunique()),
    }

    return df, summary_preprocessing

# DATA PREPARATION UNTUK FP-GROWTH

def prepare_transaction_dataset(
    df_clean: pd.DataFrame,
    kanal_filter: str = "semua",
    include_operator: bool = True,
    include_waktu: bool = True,
):
    df = df_clean.copy()

    kanal_filter = str(kanal_filter).strip().lower()

    if kanal_filter not in ["semua", "offline", "online"]:
        kanal_filter = "semua"

    if kanal_filter != "semua":
        df = df[df["kanal_penjualan"] == kanal_filter]

    if df.empty:
        raise ValueError(f"Tidak ada data untuk kanal penjualan: {kanal_filter}")

    df = df.dropna(subset=["no_transaksi", "detail_produk"])

    df["no_transaksi"] = df["no_transaksi"].apply(_safe_text)
    df["detail_produk"] = df["detail_produk"].apply(_safe_text)

    df = df.dropna(subset=["no_transaksi", "detail_produk"])

    df["item_produk"] = df["detail_produk"].apply(
        lambda x: _make_token("produk_", x)
    )

    item_columns = ["item_produk"]

    if include_operator and "operator" in df.columns:
        df["operator"] = df["operator"].apply(_safe_text)
        df["item_operator"] = df["operator"].apply(
            lambda x: _make_token("operator_", x)
        )
        item_columns.append("item_operator")

    if include_waktu and "kategori_waktu" in df.columns:
        df["kategori_waktu"] = df["kategori_waktu"].apply(_safe_text)
        df["item_waktu"] = df["kategori_waktu"].apply(
            lambda x: _make_token("waktu_", x)
        )
        item_columns.append("item_waktu")

    for col in item_columns:
        df[col] = df[col].apply(_safe_text)

    basket_records = []

    for no_transaksi, group in df.groupby("no_transaksi"):
        items = []

        for col in item_columns:
            for value in group[col].tolist():
                text = _safe_text(value)

                if text is not None:
                    items.append(text)

        itemset = list(dict.fromkeys(items))

        if len(itemset) > 0:
            basket_records.append({
                "no_transaksi": str(no_transaksi),
                "itemset": itemset,
                "jumlah_item": len(itemset),
            })

    basket = pd.DataFrame(basket_records)

    if basket.empty:
        raise ValueError("Tidak ada transaksi valid setelah proses itemset.")

    te = TransactionEncoder()
    te_data = te.fit(basket["itemset"].tolist()).transform(basket["itemset"].tolist())

    df_fp = pd.DataFrame(te_data, columns=[str(col) for col in te.columns_])

    summary_kanal = {
        "kanal_filter": kanal_filter,
        "jumlah_data_setelah_filter_kanal": int(df.shape[0]),
        "jumlah_transaksi_setelah_filter_kanal": int(df["no_transaksi"].nunique()),
    }

    return basket, df_fp, df, summary_kanal


# =========================================================
# FP-GROWTH + ASSOCIATION RULES
# =========================================================


def _empty_channel_result(
    kanal_filter: str,
    summary_preprocessing: Dict[str, Any],
    message: str,
    min_support: float,
    min_confidence: float,
    min_lift: float,
):
    empty_frequent = pd.DataFrame(
        columns=["kanal_filter", "itemsets_raw", "itemsets_display", "support", "item_count"]
    )

    empty_rules = pd.DataFrame(
        columns=[
            "kanal_filter",
            "antecedents_raw",
            "consequents_raw",
            "antecedents_display",
            "consequents_display",
            "support",
            "confidence",
            "lift",
            "kategori_rule",
            "is_anomaly",
        ]
    )

    empty_produk = pd.DataFrame(columns=["kanal_filter", "nama_produk", "jumlah_terjual"])
    empty_waktu = pd.DataFrame(columns=["kanal_filter", "kategori_waktu", "persentase"])
    empty_kanal = pd.DataFrame(columns=["kanal_filter", "kanal_penjualan", "persentase"])
    empty_operator_waktu = pd.DataFrame(columns=["kanal_filter", "operator", "kategori_waktu", "jumlah", "persentase"])
    empty_distribusi_rule = pd.DataFrame(columns=["kanal_filter", "kategori_rule", "jumlah"])

    summary = {
        **summary_preprocessing,
        "kanal_filter": kanal_filter,
        "jumlah_data_setelah_filter_kanal": 0,
        "jumlah_transaksi_setelah_filter_kanal": 0,
        "total_basket": 0,
        "produk_unik": 0,
        "operator_unik": 0,
        # Pola sering muncul = frequent itemsets.
        "frequent_itemsets": 0,
        "pola_sering_muncul": 0,
        "total_pola_sering_muncul": 0,

        # Pola hubungan = association rules yang lolos filter min_confidence dan min_lift.
        "association_rules_total": 0,
        "association_rules": 0,
        "pola_hubungan_total": 0,
        "pola_hubungan": 0,
        "jumlah_anomali": 0,
        "rule_terbaik": None,
        "min_support": float(min_support),
        "min_confidence": float(min_confidence),
        "min_lift": float(min_lift),
        "message": message,
    }

    return {
        "kanal_filter": kanal_filter,
        "summary": summary,
        "basket": pd.DataFrame(columns=["no_transaksi", "itemset", "jumlah_item"]),
        "df_fp": pd.DataFrame(),
        "df_analysis": pd.DataFrame(),
        "frequent_export": empty_frequent,
        "rules_all_export": empty_rules,
        "rules_export": empty_rules,
        "rules_anomaly_export": empty_rules,
        "produk_rekap": empty_produk,
        "waktu_dist": empty_waktu,
        "kanal_dist": empty_kanal,
        "operator_waktu": empty_operator_waktu,
        "distribusi_rule": empty_distribusi_rule,
    }


def _run_single_channel_analysis(
    df_clean: pd.DataFrame,
    summary_preprocessing: Dict[str, Any],
    kanal_filter: str,
    min_support: float,
    min_confidence: float,
    min_lift: float,
    include_operator: bool,
    include_waktu: bool,
    only_product_rules: bool,
):
    try:
        basket, df_fp, df_analysis, summary_kanal = prepare_transaction_dataset(
            df_clean,
            kanal_filter=kanal_filter,
            include_operator=include_operator,
            include_waktu=include_waktu,
        )
    except ValueError as e:
        return _empty_channel_result(
            kanal_filter=kanal_filter,
            summary_preprocessing=summary_preprocessing,
            message=str(e),
            min_support=min_support,
            min_confidence=min_confidence,
            min_lift=min_lift,
        )

    frequent_itemsets = fpgrowth(
        df_fp,
        min_support=min_support,
        use_colnames=True
    )

    if frequent_itemsets.empty:
        result = _empty_channel_result(
            kanal_filter=kanal_filter,
            summary_preprocessing=summary_preprocessing,
            message="Tidak ada frequent itemset. Coba turunkan min_support.",
            min_support=min_support,
            min_confidence=min_confidence,
            min_lift=min_lift,
        )
        result["basket"] = basket
        result["df_fp"] = df_fp
        result["df_analysis"] = df_analysis
        result["summary"] = {
            **summary_preprocessing,
            **summary_kanal,
            "total_basket": int(basket.shape[0]),
            "produk_unik": int(df_analysis["detail_produk"].nunique()),
            "operator_unik": int(df_analysis["operator"].nunique()),
            "frequent_itemsets": 0,
            "association_rules_total": 0,
            "association_rules": 0,
            "jumlah_anomali": 0,
            "rule_terbaik": None,
            "min_support": float(min_support),
            "min_confidence": float(min_confidence),
            "min_lift": float(min_lift),
            "message": "Tidak ada frequent itemset. Coba turunkan min_support.",
        }
        return result

    frequent_itemsets["item_count"] = frequent_itemsets["itemsets"].apply(len)
    frequent_itemsets["kanal_filter"] = kanal_filter

    frequent_itemsets = frequent_itemsets.sort_values(
        by=["support", "item_count"],
        ascending=[False, False]
    ).reset_index(drop=True)

    rules = association_rules(
        frequent_itemsets.drop(columns=["kanal_filter"]),
        metric="confidence",
        min_threshold=min_confidence
    )

    if not rules.empty:
        rules["kanal_filter"] = kanal_filter

        # Deteksi kategori rule dan anomali dilakukan sebelum filter lift,
        # mengikuti alur model.py.
        rules = add_rule_category_and_anomaly(rules)

        rules = rules.sort_values(
            by=["lift", "confidence", "support"],
            ascending=[False, False, False]
        ).reset_index(drop=True)

    rules_all = rules.copy()

    if not rules.empty:
        rules_filtered = rules[
            (rules["lift"] > min_lift) &
            (rules["confidence"] >= min_confidence)
        ].copy()
    else:
        rules_filtered = rules.copy()

    if only_product_rules and not rules_filtered.empty:
        rules_filtered = rules_filtered[
            rules_filtered["consequents"].apply(
                lambda itemset: all(str(item).startswith("produk_") for item in itemset)
            )
        ].copy()

    if not rules_filtered.empty:
        rules_filtered = rules_filtered.sort_values(
            by=["lift", "confidence", "support"],
            ascending=[False, False, False]
        ).reset_index(drop=True)

    frequent_export = frequent_itemsets.copy()
    frequent_export["itemsets_raw"] = frequent_export["itemsets"].apply(_itemset_to_text)
    frequent_export["itemsets_display"] = frequent_export["itemsets"].apply(_itemset_to_display)

    frequent_export = frequent_export[
        [
            "kanal_filter",
            "itemsets_raw",
            "itemsets_display",
            "support",
            "item_count",
        ]
    ]

    rules_all_export = build_rules_export(rules_all)
    rules_export = build_rules_export(rules_filtered)

    if not rules_all.empty:
        rules_anomaly = rules_all[rules_all["is_anomaly"] == True].copy()
        rules_anomaly = rules_anomaly.sort_values(
            by=["lift", "confidence", "support"],
            ascending=[True, True, False]
        ).reset_index(drop=True)
    else:
        rules_anomaly = pd.DataFrame()

    rules_anomaly_export = build_rules_export(rules_anomaly)

    rule_terbaik = None

    # Rule terbaik untuk tampilan ringkasan hanya boleh mengambil rule normal,
    # bukan rule anomali/outlier, meskipun lift rule anomali lebih tinggi.
    if not rules_export.empty:
        normal_rules_export = rules_export[
            rules_export["is_anomaly"].apply(lambda value: bool(value) is False)
        ].copy()

        if not normal_rules_export.empty:
            best_rule = normal_rules_export.sort_values(
                by=["lift", "confidence", "support"],
                ascending=[False, False, False]
            ).iloc[0]

            rule_terbaik = (
                f"{best_rule['antecedents_display']} → "
                f"{best_rule['consequents_display']}"
            )

    produk_rekap = (
        df_analysis["detail_produk"]
        .value_counts()
        .reset_index()
    )
    produk_rekap.columns = ["nama_produk", "jumlah_terjual"]
    produk_rekap["kanal_filter"] = kanal_filter
    produk_rekap = produk_rekap[
        ["kanal_filter", "nama_produk", "jumlah_terjual"]
    ]

    waktu_dist = (
        df_analysis["kategori_waktu"]
        .value_counts(normalize=True, dropna=True)
        .mul(100)
        .reset_index()
    )
    waktu_dist.columns = ["kategori_waktu", "persentase"]
    waktu_dist["kanal_filter"] = kanal_filter
    waktu_dist = waktu_dist[
        ["kanal_filter", "kategori_waktu", "persentase"]
    ]

    kanal_dist = (
        df_analysis["kanal_penjualan"]
        .value_counts(normalize=True, dropna=True)
        .mul(100)
        .reset_index()
    )
    kanal_dist.columns = ["kanal_penjualan", "persentase"]
    kanal_dist["kanal_filter"] = kanal_filter
    kanal_dist = kanal_dist[
        ["kanal_filter", "kanal_penjualan", "persentase"]
    ]

    operator_waktu = (
        df_analysis
        .groupby(["operator", "kategori_waktu"])
        .size()
        .reset_index(name="jumlah")
    )

    if not operator_waktu.empty:
        operator_waktu["persentase"] = operator_waktu.groupby("operator")["jumlah"].transform(
            lambda x: x / x.sum() * 100
        )

    operator_waktu["kanal_filter"] = kanal_filter
    operator_waktu = operator_waktu[
        ["kanal_filter", "operator", "kategori_waktu", "jumlah", "persentase"]
    ]

    if not rules_all.empty and "kategori_rule" in rules_all.columns:
        distribusi_rule = (
            rules_all
            .groupby(["kanal_filter", "kategori_rule"])
            .size()
            .reset_index(name="jumlah")
        )
    else:
        distribusi_rule = pd.DataFrame(columns=["kanal_filter", "kategori_rule", "jumlah"])

    summary = {
        **summary_preprocessing,
        **summary_kanal,
        "total_basket": int(basket.shape[0]),
        "produk_unik": int(df_analysis["detail_produk"].nunique()),
        "operator_unik": int(df_analysis["operator"].nunique()),
        # Pola sering muncul = frequent itemsets.
        "frequent_itemsets": int(frequent_export.shape[0]),
        "pola_sering_muncul": int(frequent_export.shape[0]),
        "total_pola_sering_muncul": int(frequent_export.shape[0]),

        # Pola hubungan = association rules yang lolos filter min_confidence dan min_lift.
        # association_rules_total tetap disimpan sebagai total rules sebelum filter min_lift.
        "association_rules_total": int(rules_all_export.shape[0]),
        "association_rules": int(rules_export.shape[0]),
        "pola_hubungan_total": int(rules_all_export.shape[0]),
        "pola_hubungan": int(rules_export.shape[0]),
        "jumlah_anomali": int(rules_anomaly_export.shape[0]),
        "rule_terbaik": rule_terbaik,
        "min_support": float(min_support),
        "min_confidence": float(min_confidence),
        "min_lift": float(min_lift),
    }

    return {
        "kanal_filter": kanal_filter,
        "summary": summary,
        "basket": basket,
        "df_fp": df_fp,
        "df_analysis": df_analysis,
        "frequent_export": frequent_export,
        "rules_all_export": rules_all_export,
        "rules_export": rules_export,
        "rules_anomaly_export": rules_anomaly_export,
        "produk_rekap": produk_rekap,
        "waktu_dist": waktu_dist,
        "kanal_dist": kanal_dist,
        "operator_waktu": operator_waktu,
        "distribusi_rule": distribusi_rule,
    }


def _concat_dataframes(dataframes: List[pd.DataFrame], columns: List[str]) -> pd.DataFrame:
    valid_dataframes = [
        df for df in dataframes
        if isinstance(df, pd.DataFrame) and not df.empty
    ]

    if len(valid_dataframes) == 0:
        return pd.DataFrame(columns=columns)

    return pd.concat(valid_dataframes, ignore_index=True)


def run_fpgrowth_analysis(
    file_paths: Union[str, List[str]],
    min_support: float = DEFAULT_MIN_SUPPORT,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_lift: float = DEFAULT_MIN_LIFT,
    kanal_filter: str = "semua",
    include_operator: bool = True,
    include_waktu: bool = True,
    only_product_rules: bool = False,
    top_n: int = 0,
    output_dir: str = "output/api_result",
    save_output: bool = True,
    save_intermediate: bool = False,
) -> Dict[str, Any]:

    if not (0 < min_support <= 1):
        raise ValueError("min_support harus lebih dari 0 dan maksimal 1.")

    if not (0 < min_confidence <= 1):
        raise ValueError("min_confidence harus lebih dari 0 dan maksimal 1.")

    if min_lift < 0:
        raise ValueError("min_lift tidak boleh kurang dari 0.")

    # top_n <= 0 berarti tidak ada pembatasan jumlah data yang dikirim.
    # Untuk kebutuhan aplikasi terbaru, semua association rules harus dikirim ke Laravel.
    try:
        top_n = int(top_n)
    except (TypeError, ValueError):
        top_n = 0

    os.makedirs(output_dir, exist_ok=True)

    run_id = uuid.uuid4().hex[:8]

    df_raw = _read_excel_files(file_paths)

    df_clean, summary_preprocessing = preprocess_raw_dataframe(df_raw)

    if df_clean.empty:
        raise ValueError("Dataset kosong setelah preprocessing.")

    daftar_kanal = ["offline", "online"]

    channel_results = []

    for kanal in daftar_kanal:
        channel_result = _run_single_channel_analysis(
            df_clean=df_clean,
            summary_preprocessing=summary_preprocessing,
            kanal_filter=kanal,
            min_support=min_support,
            min_confidence=min_confidence,
            min_lift=min_lift,
            include_operator=include_operator,
            include_waktu=include_waktu,
            only_product_rules=only_product_rules,
        )

        channel_results.append(channel_result)

    frequent_export_all = _concat_dataframes(
        [result["frequent_export"] for result in channel_results],
        ["kanal_filter", "itemsets_raw", "itemsets_display", "support", "item_count"]
    )

    rules_all_export = _concat_dataframes(
        [result["rules_all_export"] for result in channel_results],
        [
            "kanal_filter",
            "antecedents_raw",
            "consequents_raw",
            "antecedents_display",
            "consequents_display",
            "support",
            "confidence",
            "lift",
            "kategori_rule",
            "is_anomaly",
        ]
    )

    rules_export = _concat_dataframes(
        [result["rules_export"] for result in channel_results],
        [
            "kanal_filter",
            "antecedents_raw",
            "consequents_raw",
            "antecedents_display",
            "consequents_display",
            "support",
            "confidence",
            "lift",
            "kategori_rule",
            "is_anomaly",
        ]
    )

    rules_anomaly_export = _concat_dataframes(
        [result["rules_anomaly_export"] for result in channel_results],
        [
            "kanal_filter",
            "antecedents_raw",
            "consequents_raw",
            "antecedents_display",
            "consequents_display",
            "support",
            "confidence",
            "lift",
            "kategori_rule",
            "is_anomaly",
        ]
    )

    produk_rekap_all = _concat_dataframes(
        [result["produk_rekap"] for result in channel_results],
        ["kanal_filter", "nama_produk", "jumlah_terjual"]
    )

    waktu_dist_all = _concat_dataframes(
        [result["waktu_dist"] for result in channel_results],
        ["kanal_filter", "kategori_waktu", "persentase"]
    )

    kanal_dist_all = _concat_dataframes(
        [result["kanal_dist"] for result in channel_results],
        ["kanal_filter", "kanal_penjualan", "persentase"]
    )

    operator_waktu_all = _concat_dataframes(
        [result["operator_waktu"] for result in channel_results],
        ["kanal_filter", "operator", "kategori_waktu", "jumlah", "persentase"]
    )

    distribusi_rule_all = _concat_dataframes(
        [result["distribusi_rule"] for result in channel_results],
        ["kanal_filter", "kategori_rule", "jumlah"]
    )

    if not rules_export.empty:
        rules_export = rules_export.sort_values(
            by=["kanal_filter", "lift", "confidence", "support"],
            ascending=[True, False, False, False]
        ).reset_index(drop=True)

    if not rules_all_export.empty:
        rules_all_export = rules_all_export.sort_values(
            by=["kanal_filter", "lift", "confidence", "support"],
            ascending=[True, False, False, False]
        ).reset_index(drop=True)

    if not rules_anomaly_export.empty:
        rules_anomaly_export = rules_anomaly_export.sort_values(
            by=["kanal_filter", "lift", "confidence", "support"],
            ascending=[True, True, True, False]
        ).reset_index(drop=True)

    # Untuk aplikasi Laravel, field top_rules tetap dipakai sebagai sumber utama data tabel.
    # Karena itu jangan dipotong top_n. Kirim semua rules yang lolos filter support,
    # confidence, lift, dan aturan only_product_rules.
    if not rules_export.empty:
        top_rules_all = rules_export.sort_values(
            by=["kanal_filter", "lift", "confidence", "support"],
            ascending=[True, False, False, False]
        ).reset_index(drop=True)
    else:
        top_rules_all = pd.DataFrame(columns=rules_export.columns)

    top_frequent_per_kanal = []

    for kanal in daftar_kanal:
        frequent_kanal = frequent_export_all[frequent_export_all["kanal_filter"] == kanal].copy()

        if not frequent_kanal.empty:
            frequent_kanal = frequent_kanal.sort_values(
                by=["support", "item_count"],
                ascending=[False, False]
            )

            if top_n > 0:
                frequent_kanal = frequent_kanal.head(top_n)

            top_frequent_per_kanal.append(frequent_kanal)

    if len(top_frequent_per_kanal) > 0:
        top_frequent_all = pd.concat(top_frequent_per_kanal, ignore_index=True)
    else:
        top_frequent_all = pd.DataFrame(columns=frequent_export_all.columns)

    # Rules anomali juga tidak dipotong, supaya mode anomali di Laravel bisa melihat semuanya.
    if not rules_anomaly_export.empty:
        anomaly_rules_all = rules_anomaly_export.sort_values(
            by=["kanal_filter", "lift", "confidence", "support"],
            ascending=[True, True, True, False]
        ).reset_index(drop=True)
    else:
        anomaly_rules_all = pd.DataFrame(columns=rules_anomaly_export.columns)

    channel_summary = {
        result["kanal_filter"]: result["summary"]
        for result in channel_results
    }

    summary_offline = channel_summary.get("offline", {})
    summary_online = channel_summary.get("online", {})

    rule_terbaik = None

    # Rule terbaik gabungan juga hanya boleh dari rule normal.
    if not rules_export.empty:
        normal_rules_export = rules_export[
            rules_export["is_anomaly"].apply(lambda value: bool(value) is False)
        ].copy()

        if not normal_rules_export.empty:
            best_rule = normal_rules_export.sort_values(
                by=["lift", "confidence", "support"],
                ascending=[False, False, False]
            ).iloc[0]

            rule_terbaik = (
                f"{best_rule['antecedents_display']} → "
                f"{best_rule['consequents_display']}"
            )

    total_basket_gabungan = (
        int(summary_offline.get("total_basket", 0)) +
        int(summary_online.get("total_basket", 0))
    )

    frequent_itemsets_gabungan = (
        int(summary_offline.get("frequent_itemsets", 0)) +
        int(summary_online.get("frequent_itemsets", 0))
    )

    association_rules_total_gabungan = (
        int(summary_offline.get("association_rules_total", 0)) +
        int(summary_online.get("association_rules_total", 0))
    )

    pola_hubungan_gabungan = int(rules_export.shape[0])

    jumlah_anomali_gabungan = (
        int(summary_offline.get("jumlah_anomali", 0)) +
        int(summary_online.get("jumlah_anomali", 0))
    )

    output_files = {}

    if save_output:
        clean_path = os.path.join(output_dir, f"{run_id}_dataset_bersih.xlsx")
        frequent_path = os.path.join(output_dir, f"{run_id}_frequent_itemsets.xlsx")
        rules_all_path = os.path.join(output_dir, f"{run_id}_association_rules_all.xlsx")
        rules_filtered_path = os.path.join(output_dir, f"{run_id}_association_rules_filtered.xlsx")
        anomaly_path = os.path.join(output_dir, f"{run_id}_rules_anomali.xlsx")
        produk_rekap_path = os.path.join(output_dir, f"{run_id}_rekap_produk.xlsx")
        waktu_dist_path = os.path.join(output_dir, f"{run_id}_distribusi_waktu.xlsx")
        kanal_dist_path = os.path.join(output_dir, f"{run_id}_distribusi_kanal.xlsx")
        operator_waktu_path = os.path.join(output_dir, f"{run_id}_operator_waktu.xlsx")
        distribusi_rule_path = os.path.join(output_dir, f"{run_id}_distribusi_rule.xlsx")

        df_clean.to_excel(clean_path, index=False)
        frequent_export_all.to_excel(frequent_path, index=False)
        rules_all_export.to_excel(rules_all_path, index=False)
        rules_export.to_excel(rules_filtered_path, index=False)
        rules_anomaly_export.to_excel(anomaly_path, index=False)
        produk_rekap_all.to_excel(produk_rekap_path, index=False)
        waktu_dist_all.to_excel(waktu_dist_path, index=False)
        kanal_dist_all.to_excel(kanal_dist_path, index=False)
        operator_waktu_all.to_excel(operator_waktu_path, index=False)
        distribusi_rule_all.to_excel(distribusi_rule_path, index=False)

        output_files = {
            "dataset_bersih": clean_path,
            "frequent_itemsets": frequent_path,
            "association_rules_all": rules_all_path,
            "association_rules": rules_filtered_path,
            "association_rules_filtered": rules_filtered_path,
            "rules_anomali": anomaly_path,
            "rekap_produk": produk_rekap_path,
            "distribusi_waktu": waktu_dist_path,
            "distribusi_kanal": kanal_dist_path,
            "operator_waktu": operator_waktu_path,
            "distribusi_rule": distribusi_rule_path,
        }

        if save_intermediate:
            for result in channel_results:
                kanal = result["kanal_filter"]

                basket_path = os.path.join(output_dir, f"{run_id}_dataset_transaksi_itemset_{kanal}.xlsx")
                fp_path = os.path.join(output_dir, f"{run_id}_dataset_fp_growth_{kanal}.xlsx")

                basket_export = result["basket"].copy()

                if not basket_export.empty and "itemset" in basket_export.columns:
                    basket_export["itemset"] = basket_export["itemset"].apply(_safe_join)

                basket_export.to_excel(basket_path, index=False)
                result["df_fp"].to_excel(fp_path, index=False)

                output_files[f"dataset_transaksi_itemset_{kanal}"] = basket_path
                output_files[f"dataset_fp_growth_{kanal}"] = fp_path

    result = {
        "status": "success",
        "message": "Analisis FP-Growth berhasil diproses untuk kanal offline dan online.",
        "summary": {
            **summary_preprocessing,
            "kanal_filter": "semua",
            "kanal_filter_label": "Semua Kanal",
            "keterangan_kanal": "Semua Kanal berarti gabungan rules offline dan online, bukan hasil mining kanal tersendiri.",
            "total_basket": int(total_basket_gabungan),
            "produk_unik": int(df_clean["detail_produk"].nunique()),
            "operator_unik": int(df_clean["operator"].nunique()),
            # Pola sering muncul = frequent itemsets.
            "frequent_itemsets": int(frequent_itemsets_gabungan),
            "pola_sering_muncul": int(frequent_itemsets_gabungan),
            "total_pola_sering_muncul": int(frequent_itemsets_gabungan),
            "pola_sering_muncul_offline": int(summary_offline.get("frequent_itemsets", 0)),
            "pola_sering_muncul_online": int(summary_online.get("frequent_itemsets", 0)),

            # Pola hubungan = association rules yang lolos filter min_confidence dan min_lift.
            "association_rules_total": int(association_rules_total_gabungan),
            "association_rules": int(pola_hubungan_gabungan),
            "pola_hubungan_total": int(association_rules_total_gabungan),
            "pola_hubungan": int(pola_hubungan_gabungan),
            "association_rules_offline": int(summary_offline.get("association_rules", 0)),
            "association_rules_online": int(summary_online.get("association_rules", 0)),
            "pola_hubungan_offline": int(summary_offline.get("association_rules", 0)),
            "pola_hubungan_online": int(summary_online.get("association_rules", 0)),
            "jumlah_anomali": int(jumlah_anomali_gabungan),
            "jumlah_anomali_offline": int(summary_offline.get("jumlah_anomali", 0)),
            "jumlah_anomali_online": int(summary_online.get("jumlah_anomali", 0)),
            "rule_terbaik": rule_terbaik,
            "min_support": float(min_support),
            "min_confidence": float(min_confidence),
            "min_lift": float(min_lift),
        },
        "summary_per_kanal": channel_summary,
        "top_frequent_itemsets": _df_to_records(
            top_frequent_all
        ),
        "top_rules": _df_to_records(
            top_rules_all
        ),
        # Alias tambahan supaya aman jika controller membaca key lain.
        "rules": _df_to_records(
            top_rules_all
        ),
        "association_rules_data": _df_to_records(
            top_rules_all
        ),
        "anomaly_rules": _df_to_records(
            anomaly_rules_all
        ),
        "rekap_produk": _df_to_records(
            produk_rekap_all
        ),
        "distribusi_waktu": _df_to_records(
            waktu_dist_all
        ),
        "distribusi_kanal": _df_to_records(
            kanal_dist_all
        ),
        "distribusi_rule": _df_to_records(
            distribusi_rule_all
        ),
        "output_files": output_files,
    }

    return result
