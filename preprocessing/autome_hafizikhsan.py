from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR.parent / "loan_approval_raw.csv"
OUTPUT_PATH = BASE_DIR / "loan_preprocessed.csv"


def preprocessing_data(df: pd.DataFrame):

    df = df.drop(columns=["name", "points"])

    X = df.drop("loan_approved", axis=1)
    y = df["loan_approved"]

    le = LabelEncoder()
    y = le.fit_transform(y)

    X = pd.get_dummies(X, columns=["city"])
    df_processed = pd.concat([X, pd.Series(y, name="loan_approved")], axis=1)

    return df_processed


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    df_processed = preprocessing_data(df)
    df_processed.to_csv(OUTPUT_PATH, index=False)
    print(f"Preprocessing selesai. File tersimpan di: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()


