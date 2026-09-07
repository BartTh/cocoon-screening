"""Validation shared by annotation UI and session state."""
import pandas as pd


def validate_classifications(table, class_count):
    if not {"instance_id", "class_index"}.issubset(table.columns):
        raise ValueError("Classification CSV needs instance_id and class_index columns.")
    table = table.copy()
    for column in ("instance_id", "class_index"):
        table[column] = pd.to_numeric(table[column], errors="raise").astype("Int64")
    ids = table["instance_id"]
    if ids.isna().any() or (ids <= 0).any() or ids.duplicated().any():
        raise ValueError("Instance IDs must be unique positive integers.")
    classes = table["class_index"].dropna()
    if ((classes < 0) | (classes >= class_count)).any():
        raise ValueError(f"Class indices must be between 0 and {class_count - 1}.")
    return table


def classification_progress(table, instance_ids, class_count):
    ids = set(instance_ids)
    try:
        validated = validate_classifications(table, class_count)
    except (ValueError, TypeError, OverflowError):
        return 0, False
    matched = validated[validated["instance_id"].isin(ids)]
    count = int(matched["class_index"].notna().sum())
    complete = set(validated["instance_id"]) == ids and count == len(ids)
    return count, complete
