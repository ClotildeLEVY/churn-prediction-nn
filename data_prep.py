import pandas as pd

def prepare_data(df):
    # Préparation des données : rééquilibrage des classes
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    df_majority = df[df["Churn"] == 0]
    df_minority = df[df["Churn"] == 1]
    df_minority_oversampled = df_minority.sample(n=len(df_majority), replace=True, random_state=42)
    df_balanced = pd.concat([df_majority, df_minority_oversampled])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    return df_balanced
