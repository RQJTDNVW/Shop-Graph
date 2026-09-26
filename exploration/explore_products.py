import pandas as pd
import numpy as np
import json
import os


# ============================================================
# SHOPGRAPH - PRODUCT DATA EXPLORATION
# ============================================================

DATA_FILE = "data/processed/products_100k.parquet"
OUTPUT_DIR = "data/processed"

REPORT_FILE = os.path.join(
    OUTPUT_DIR,
    "exploration_report.json"
)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 75)
print("SHOPGRAPH - PRODUCT DATA EXPLORATION")
print("=" * 75)

print("\nLoading dataset...")

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}"
    )

df = pd.read_parquet(DATA_FILE)

print("Dataset loaded successfully!")

print("\nShape:")
print(df.shape)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_is_missing(value):
    """
    Safely determine whether a value is missing.

    Handles:
    - None
    - NaN
    - lists
    - NumPy arrays
    - strings
    - other objects
    """

    if value is None:
        return True

    if isinstance(value, np.ndarray):
        return value.size == 0

    if isinstance(value, (list, tuple)):
        return len(value) == 0

    try:
        result = pd.isna(value)

        if isinstance(result, (bool, np.bool_)):
            return bool(result)

        return False

    except (TypeError, ValueError):
        return False


def list_to_text(value):
    """
    Convert list/array/string/object into clean text.
    """

    if value is None:
        return ""

    # Python list
    if isinstance(value, list):
        return " ".join(
            str(item)
            for item in value
            if item is not None
        ).strip()

    # Tuple
    if isinstance(value, tuple):
        return " ".join(
            str(item)
            for item in value
            if item is not None
        ).strip()

    # NumPy array
    if isinstance(value, np.ndarray):
        try:
            items = value.tolist()

            if isinstance(items, list):
                return " ".join(
                    str(item)
                    for item in items
                    if item is not None
                ).strip()

            return str(items).strip()

        except Exception:
            return str(value).strip()

    # String
    if isinstance(value, str):
        return value.strip()

    # Other scalar values
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def count_list_items(value):
    """
    Count elements inside list-like fields.
    """

    if value is None:
        return 0

    if isinstance(value, (list, tuple)):
        return len(value)

    if isinstance(value, np.ndarray):
        return value.size

    return 0


def image_count(value):
    """
    Count available large product images.
    """

    if not isinstance(value, dict):
        return 0

    images = value.get("large", [])

    if isinstance(images, np.ndarray):
        images = images.tolist()

    if not isinstance(images, list):
        return 0

    return sum(
        1
        for image in images
        if image is not None
        and str(image).strip() != ""
    )


def first_category(value):
    """
    Return the first category safely.
    """

    if isinstance(value, (list, tuple)):
        if len(value) > 0:
            return str(value[0])

    if isinstance(value, np.ndarray):
        if value.size > 0:
            return str(value[0])

    return "Unknown"


# ============================================================
# 1. BASIC INFORMATION
# ============================================================

print("\n" + "=" * 75)
print("1. BASIC DATASET INFORMATION")
print("=" * 75)

print("\nNumber of products:", len(df))
print("Number of columns:", len(df.columns))

print("\nColumns:")

for i, column in enumerate(df.columns, start=1):
    print(f"{i:2}. {column}")


# ============================================================
# 2. DATA TYPES
# ============================================================

print("\n" + "=" * 75)
print("2. DATA TYPES")
print("=" * 75)

print(df.dtypes)


# ============================================================
# 3. MISSING VALUES
# ============================================================

print("\n" + "=" * 75)
print("3. MISSING VALUES")
print("=" * 75)

missing_count = df.isna().sum()

missing_percent = (
    missing_count / len(df)
) * 100

missing_table = pd.DataFrame({
    "missing_count": missing_count,
    "missing_percent": missing_percent.round(2)
})

missing_table = missing_table.sort_values(
    "missing_percent",
    ascending=False
)

print(
    missing_table.to_string()
)


# ============================================================
# 4. EMPTY VALUES
# ============================================================

print("\n" + "=" * 75)
print("4. EMPTY VALUES")
print("=" * 75)

text_columns = [
    "title",
    "store",
    "subtitle",
    "author"
]

empty_value_results = {}

for column in text_columns:

    if column in df.columns:

        empty_count = (
            df[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )

        percentage = (
            empty_count / len(df)
        ) * 100

        empty_value_results[column] = {
            "count": int(empty_count),
            "percentage": float(percentage)
        }

        print(
            f"{column:20} "
            f"empty: {empty_count:,} "
            f"({percentage:.2f}%)"
        )


# ============================================================
# 5. PRODUCT TITLE ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print("5. PRODUCT TITLE ANALYSIS")
print("=" * 75)

title_text = (
    df["title"]
    .fillna("")
    .astype(str)
    .str.strip()
)

title_length = title_text.str.len()

print("\nTitle length statistics:")

print(
    title_length
    .describe()
    .to_string()
)

no_title = (
    title_text
    .eq("")
    .sum()
)

print(
    "\nProducts without usable title:",
    f"{no_title:,}"
)

print(
    "Percentage without usable title:",
    f"{no_title / len(df) * 100:.2f}%"
)


# ============================================================
# 6. DESCRIPTION ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print("6. DESCRIPTION ANALYSIS")
print("=" * 75)

description_text = (
    df["description"]
    .apply(list_to_text)
)

description_length = (
    description_text
    .str.len()
)

print("\nDescription length statistics:")

print(
    description_length
    .describe()
    .to_string()
)

empty_description = (
    description_text
    .str.strip()
    .eq("")
    .sum()
)

print(
    "\nProducts without description:",
    f"{empty_description:,}"
)

print(
    "Percentage without description:",
    f"{empty_description / len(df) * 100:.2f}%"
)


# ============================================================
# 7. FEATURES ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print("7. PRODUCT FEATURES ANALYSIS")
print("=" * 75)

feature_count = (
    df["features"]
    .apply(count_list_items)
)

print("\nFeatures per product:")

print(
    feature_count
    .describe()
    .to_string()
)

no_features = (
    feature_count == 0
).sum()

print(
    "\nProducts without features:",
    f"{no_features:,}"
)

print(
    "Percentage without features:",
    f"{no_features / len(df) * 100:.2f}%"
)


# ============================================================
# 8. CATEGORY ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print("8. CATEGORY ANALYSIS")
print("=" * 75)

category_count = (
    df["categories"]
    .apply(count_list_items)
)

print("\nCategories per product:")

print(
    category_count
    .describe()
    .to_string()
)

no_categories = (
    category_count == 0
).sum()

print(
    "\nProducts without categories:",
    f"{no_categories:,}"
)

print(
    "Percentage without categories:",
    f"{no_categories / len(df) * 100:.2f}%"
)


print("\nTop-level categories:")

top_categories = (
    df["categories"]
    .apply(first_category)
    .value_counts()
    .head(20)
)

print(
    top_categories
    .to_string()
)


# ============================================================
# 9. STORE / BRAND ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print("9. STORE / BRAND ANALYSIS")
print("=" * 75)

store_count = df["store"].nunique(
    dropna=True
)

print(
    "\nUnique stores/brands:",
    store_count
)

print("\nTop 20 stores/brands:")

top_stores = (
    df["store"]
    .fillna("Unknown")
    .astype(str)
    .str.strip()
    .replace("", "Unknown")
    .value_counts()
    .head(20)
)

print(
    top_stores
    .to_string()
)


# ============================================================
# 10. RATING ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print("10. RATING ANALYSIS")
print("=" * 75)

print("\nAverage rating statistics:")

print(
    df["average_rating"]
    .describe()
    .to_string()
)

print("\nRating distribution:")

rating_distribution = (
    df["average_rating"]
    .value_counts()
    .sort_index()
)

print(
    rating_distribution
    .to_string()
)


# ============================================================
# 11. REVIEW COUNT ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print("11. REVIEW COUNT ANALYSIS")
print("=" * 75)

print("\nRating number statistics:")

print(
    df["rating_number"]
    .describe()
    .to_string()
)


# ============================================================
# 12. PRICE ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print("12. PRICE ANALYSIS")
print("=" * 75)

print("\nRaw price examples:")

price_examples = (
    df["price"]
    .dropna()
    .astype(str)
    .head(20)
)

print(
    price_examples
    .to_string(index=False)
)

price_text = (
    df["price"]
    .fillna("")
    .astype(str)
    .str.strip()
)

price_missing = (
    price_text
    .isin([
        "",
        "None",
        "none",
        "nan",
        "NaN",
        "null",
        "NULL"
    ])
    .sum()
)

print(
    f"\nMissing/unavailable prices: "
    f"{price_missing:,}"
)

print(
    f"Percentage without price: "
    f"{price_missing / len(df) * 100:.2f}%"
)


# ============================================================
# 13. IMAGE ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print("13. IMAGE AVAILABILITY")
print("=" * 75)

image_counts = (
    df["images"]
    .apply(image_count)
)

products_with_images = (
    image_counts > 0
).sum()

products_without_images = (
    image_counts == 0
).sum()

print(
    "Products with at least one image:",
    f"{products_with_images:,}"
)

print(
    "Percentage with images:",
    f"{products_with_images / len(df) * 100:.2f}%"
)

print(
    "Products without images:",
    f"{products_without_images:,}"
)


# ============================================================
# 14. PRODUCT ID ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print("14. PRODUCT ID ANALYSIS")
print("=" * 75)

unique_asins = (
    df["parent_asin"]
    .nunique(dropna=True)
)

duplicate_asin_rows = (
    len(df) - unique_asins
)

print(
    "Unique parent ASINs:",
    unique_asins
)

print(
    "Duplicate parent ASIN rows:",
    duplicate_asin_rows
)


# ============================================================
# 15. TEXT COVERAGE FOR EMBEDDINGS
# ============================================================

print("\n" + "=" * 75)
print("15. TEXT COVERAGE FOR AI EMBEDDINGS")
print("=" * 75)

title_available = (
    title_text
    .str.strip()
    .ne("")
)

description_available = (
    description_text
    .str.strip()
    .ne("")
)

features_available = (
    feature_count > 0
)

categories_available = (
    category_count > 0
)

store_available = (
    df["store"]
    .fillna("")
    .astype(str)
    .str.strip()
    .ne("")
)

print(
    "Title available:",
    f"{title_available.sum():,} "
    f"({title_available.mean() * 100:.2f}%)"
)

print(
    "Description available:",
    f"{description_available.sum():,} "
    f"({description_available.mean() * 100:.2f}%)"
)

print(
    "Features available:",
    f"{features_available.sum():,} "
    f"({features_available.mean() * 100:.2f}%)"
)

print(
    "Categories available:",
    f"{categories_available.sum():,} "
    f"({categories_available.mean() * 100:.2f}%)"
)

print(
    "Store/brand available:",
    f"{store_available.sum():,} "
    f"({store_available.mean() * 100:.2f}%)"
)


combined_text_available = (
    title_available
    | description_available
    | features_available
    | categories_available
)

print(
    "\nProducts with at least one usable "
    "text field:",
    f"{combined_text_available.sum():,} "
    f"({combined_text_available.mean() * 100:.2f}%)"
)


# ============================================================
# 16. COMBINED TEXT LENGTH
# ============================================================

print("\n" + "=" * 75)
print("16. COMBINED TEXT ANALYSIS")
print("=" * 75)


def build_product_text(row):
    """
    Create a combined textual representation
    for exploration purposes only.
    """

    parts = []

    title = list_to_text(row["title"])

    if title:
        parts.append(title)

    description = list_to_text(
        row["description"]
    )

    if description:
        parts.append(description)

    features = list_to_text(
        row["features"]
    )

    if features:
        parts.append(features)

    categories = list_to_text(
        row["categories"]
    )

    if categories:
        parts.append(categories)

    store = list_to_text(
        row["store"]
    )

    if store:
        parts.append(store)

    return " ".join(parts).strip()


print("\nBuilding combined product text...")

combined_text = df.apply(
    build_product_text,
    axis=1
)

combined_text_length = (
    combined_text
    .str.len()
)

print("\nCombined text length statistics:")

print(
    combined_text_length
    .describe()
    .to_string()
)


# ============================================================
# 17. SAMPLE PRODUCTS
# ============================================================

print("\n" + "=" * 75)
print("17. SAMPLE PRODUCTS")
print("=" * 75)

sample_columns = [
    "title",
    "average_rating",
    "rating_number",
    "store",
    "categories",
    "parent_asin"
]

available_columns = [
    column
    for column in sample_columns
    if column in df.columns
]

print(
    df[available_columns]
    .head(10)
    .to_string(index=False)
)


# ============================================================
# 18. DATA QUALITY SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("18. DATA QUALITY SUMMARY")
print("=" * 75)

print(
    f"\nTotal products:             {len(df):,}"
)

print(
    f"Unique product IDs:         {unique_asins:,}"
)

print(
    f"Duplicate product rows:     {duplicate_asin_rows:,}"
)

print(
    f"Missing titles:             {no_title:,}"
)

print(
    f"Missing descriptions:       {empty_description:,}"
)

print(
    f"Missing features:           {no_features:,}"
)

print(
    f"Missing categories:         {no_categories:,}"
)

print(
    f"Missing prices:             {price_missing:,}"
)

print(
    f"Products with images:       {products_with_images:,}"
)

print(
    f"Usable text products:       "
    f"{combined_text_available.sum():,}"
)


# ============================================================
# CREATE EXPLORATION REPORT
# ============================================================

report = {

    "dataset": {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(df.columns)
    },

    "missing_values": {
        str(column): {
            "count": int(missing_count[column]),
            "percentage": float(
                missing_percent[column]
            )
        }
        for column in df.columns
    },

    "empty_values": empty_value_results,

    "products": {
        "total": int(len(df)),
        "unique_parent_asin": int(unique_asins),
        "duplicate_parent_asin_rows": int(
            duplicate_asin_rows
        )
    },

    "title": {
        "mean_length": float(
            title_length.mean()
        ),
        "median_length": float(
            title_length.median()
        ),
        "min_length": int(
            title_length.min()
        ),
        "max_length": int(
            title_length.max()
        ),
        "without_usable_title": int(
            no_title
        )
    },

    "description": {
        "mean_length": float(
            description_length.mean()
        ),
        "median_length": float(
            description_length.median()
        ),
        "without_description": int(
            empty_description
        )
    },

    "features": {
        "mean_features": float(
            feature_count.mean()
        ),
        "median_features": float(
            feature_count.median()
        ),
        "without_features": int(
            no_features
        )
    },

    "categories": {
        "mean_categories": float(
            category_count.mean()
        ),
        "median_categories": float(
            category_count.median()
        ),
        "without_categories": int(
            no_categories
        ),
        "top_categories": {
            str(k): int(v)
            for k, v in top_categories.items()
        }
    },

    "stores": {
        "unique_stores": int(store_count),
        "top_stores": {
            str(k): int(v)
            for k, v in top_stores.items()
        }
    },

    "ratings": {
        "mean": float(
            df["average_rating"].mean()
        ),
        "median": float(
            df["average_rating"].median()
        ),
        "min": float(
            df["average_rating"].min()
        ),
        "max": float(
            df["average_rating"].max()
        )
    },

    "review_counts": {
        "mean": float(
            df["rating_number"].mean()
        ),
        "median": float(
            df["rating_number"].median()
        ),
        "min": int(
            df["rating_number"].min()
        ),
        "max": int(
            df["rating_number"].max()
        )
    },

    "price": {
        "unavailable": int(price_missing),
        "percentage_unavailable": float(
            price_missing / len(df) * 100
        )
    },

    "images": {
        "products_with_images": int(
            products_with_images
        ),
        "products_without_images": int(
            products_without_images
        ),
        "percentage_with_images": float(
            products_with_images / len(df) * 100
        )
    },

    "text_coverage": {
        "title_available": int(
            title_available.sum()
        ),
        "description_available": int(
            description_available.sum()
        ),
        "features_available": int(
            features_available.sum()
        ),
        "categories_available": int(
            categories_available.sum()
        ),
        "store_available": int(
            store_available.sum()
        ),
        "combined_text_available": int(
            combined_text_available.sum()
        )
    },

    "combined_text": {
        "mean_length": float(
            combined_text_length.mean()
        ),
        "median_length": float(
            combined_text_length.median()
        ),
        "min_length": int(
            combined_text_length.min()
        ),
        "max_length": int(
            combined_text_length.max()
        )
    }
}


# ============================================================
# SAVE EXPLORATION REPORT
# ============================================================

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        report,
        file,
        indent=4
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 75)
print("EXPLORATION COMPLETE")
print("=" * 75)

print("\nExploration report saved to:")

print(REPORT_FILE)

print("\nDataset remains unchanged.")

print("\nNext step:")
print(
    "Use the exploration results to design "
    "ShopGraph product preprocessing."
)

print("=" * 75)