import pandas as pd


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load CSV data safely and raise specific errors if input is invalid."""
    if not isinstance(csv_path, str) or not csv_path.strip():
        raise ValueError("csv_path must be a non-empty string.")
    if not csv_path.lower().endswith(".csv"):
        raise ValueError(f"Expected a .csv file path, got: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Loaded dataset from '{csv_path}' is empty.")
    return df


def filter_records(
    df: pd.DataFrame,
    city: str | None = None,
    email_domain: str | None = None,
    sort_by: str | None = None,
    ascending: bool = True,
) -> pd.DataFrame:
    """Filter and optionally sort records, raising explicit exceptions on invalid inputs."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    result = df.copy()

    if city is not None:
        if not isinstance(city, str) or not city.strip():
            raise ValueError("city filter must be a non-empty string.")
        if "city" not in result.columns:
            raise KeyError("Column 'city' is missing from DataFrame.")
        result = result[result["city"].astype(str).str.casefold() == city.strip().casefold()]

    if email_domain is not None:
        if not isinstance(email_domain, str) or not email_domain.strip():
            raise ValueError("email_domain filter must be a non-empty string.")
        if "email" not in result.columns:
            raise KeyError("Column 'email' is missing from DataFrame.")
        domain_clean = email_domain.strip().lstrip("@").lower()
        result = result[result["email"].astype(str).str.lower().str.endswith(f"@{domain_clean}")]

    if sort_by is not None:
        if sort_by not in result.columns:
            raise KeyError(f"Cannot sort by missing column '{sort_by}'.")
        result = result.sort_values(by=sort_by, ascending=ascending, na_position="last")

    return result


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a new DataFrame with derived columns without mutating the input."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    for col in ("first_name", "last_name", "email"):
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' is missing.")

    updated = df.copy()
    updated["full_name"] = (
        updated["first_name"].fillna("").astype(str).str.strip()
        + " "
        + updated["last_name"].fillna("").astype(str).str.strip()
    ).str.strip()
    updated["email_domain"] = updated["email"].astype(str).str.split("@").str[-1].str.lower()
    updated["name_length"] = updated["full_name"].str.len()
    return updated


def main() -> None:
    print("=== COMP1888 Lab 7: Full Data Manipulation Walkthrough ===\n")

    # 1. Load CSV and preserve an untouched baseline copy
    df_baseline = load_dataset("data/uk-500.csv")
    df_working = df_baseline.copy(deep=True)
    assert df_baseline.equals(df_working), "Initial baseline and working copies must match."

    print("[1] Dataset loaded & inspected")
    print(f"Shape: {df_working.shape}")
    print("Columns:", list(df_working.columns))
    print(df_working.head(3))
    print("-" * 50)

    # 2. Select columns, .loc (label-based), and .iloc (position-based)
    print("[2] Column selection, .loc, and .iloc")
    selected_cols = df_working[["first_name", "last_name", "city", "email"]].head(3)
    loc_slice = df_working.loc[0:2, ["first_name", "last_name", "county"]]
    iloc_slice = df_working.iloc[0:3, 0:4]
    assert len(loc_slice) == 3 and len(iloc_slice) == 3
    print("Selected columns:\n", selected_cols)
    print(".loc slice:\n", loc_slice)
    print(".iloc slice:\n", iloc_slice)
    print("-" * 50)

    # 3. Missing values & categorical value counts
    print("[3] Missing values & categorical counts")
    missing_counts = df_working.isna().sum()
    print("Missing values per column:\n", missing_counts[missing_counts > 0] if (missing_counts > 0).any() else "0 missing values across all columns.")
    if "county" in df_working.columns:
        print("\nTop 5 counties:\n", df_working["county"].value_counts(dropna=False).head(5))
    print("-" * 50)

    # 4. Derived columns while keeping baseline intact
    print("[4] Derived columns & baseline preservation")
    df_working = add_derived_columns(df_working)
    assert "full_name" not in df_baseline.columns, "Baseline must remain unmodified."
    assert (df_working["name_length"] >= 0).all(), "name_length must be non-negative."
    print(df_working[["full_name", "email_domain", "name_length"]].head(3))
    print("-" * 50)

    # 5. Filtering, sorting, and empty/boundary case verification
    print("[5] Filtering, sorting & empty selection check")
    london_sorted = filter_records(df_working, city="London", sort_by="last_name", ascending=True)
    print(f"London matches: {len(london_sorted)}")
    print(london_sorted[["full_name", "city", "email"]].head(3))

    # Boundary / empty selection check
    empty_check = filter_records(df_working, city="NonExistentCityXYZ")
    assert empty_check.empty, "Expected an empty DataFrame for non-existent city."
    print(f"Boundary check (NonExistentCityXYZ) returned {len(empty_check)} rows as expected.")
    print("-" * 50)

    # 6. Exception handling with targeted try/except blocks
    print("[6] Targeted exception handling")
    for bad_call, label in [
        (lambda: filter_records(df_working, city=""), "empty city string"),
        (lambda: filter_records(df_working, sort_by="does_not_exist"), "missing sort column"),
        (lambda: load_dataset("data/not_a_csv.txt"), "non-CSV file extension"),
    ]:
        try:
            bad_call()
        except (ValueError, KeyError, TypeError) as exc:
            print(f"Caught expected error for {label}: {type(exc).__name__}: {exc}")

    print("\n=== All Lab 7 goals completed and verified! ===")


if __name__ == "__main__":
    main()
