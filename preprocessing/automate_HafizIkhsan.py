from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR.parent / "loan_approval_raw.csv"
OUTPUT_PATH = BASE_DIR / "loan_approval_preprocessed.csv"


def preprocessing_data(df: pd.DataFrame):
    # Drop kolom yang tidak relevan
    df = df.drop(columns=["name", "points"])

    # Encoding Data
    label_encoder = LabelEncoder()

    # List kolom kategorikal yang perlu di-encode
    categorical_columns = ['city', 'loan_approved']
    
    # Encode kolom kategorikal
    for column in categorical_columns:
        df[column] = label_encoder.fit_transform(df[column])
    
    # Menyimpan hasil preprocessing ke file baru
    df.to_csv(OUTPUT_PATH, index=False)
    return df


def main():
    df = pd.read_csv(INPUT_PATH)
    preprocessing_data(df)

if __name__ == "__main__":
    main()


