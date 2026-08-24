import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.ensemble import GradientBoostingClassifier

TRAIN_FILE = "event_attendance_real_world.xlsx"
TEST_FILE = "event_attendance_test_real_world.xlsx"
OUTPUT_FILE = "attendance_predictions.xlsx"


def clean_data(df, has_target=False):

    data = df.copy()

    categorical_columns = [
        "event_type",
        "club_member",
        "event_day"
    ]

    for column in categorical_columns:
        data[column] = data[column].apply(
            lambda value:
                value.strip().lower()
                if isinstance(value, str)
                else np.nan
        )

    def extract_hour(value):

        if pd.isna(value):
            return np.nan

        if hasattr(value, "hour"):
            return value.hour

        try:
            return int(str(value).split(":")[0])
        except:
            return np.nan

    data["event_hour"] = data["event_time"].apply(extract_hour)

    data = data.drop(columns=["event_time"])


    data["previous_events_registered"] = data[
        [
            "previous_events_registered",
            "previous_events_attended"
        ]
    ].max(axis=1)


    data["previous_attendance_rate"] = (
        data["previous_events_attended"]
        /
        data["previous_events_registered"].replace(0, np.nan)
    )

    data["previous_attendance_rate"] = (
        data["previous_attendance_rate"].clip(0, 1)
    )


    if has_target:

        data = data[data["attended"].notna()].copy()

        y = data["attended"].astype(int)

        data = data.drop(columns=["attended"])

        return data, y

    return data


print("\nLoading datasets...")

train_df = pd.read_excel(TRAIN_FILE)
test_df = pd.read_excel(TEST_FILE)

print("Training dataset shape:", train_df.shape)
print("Test dataset shape:", test_df.shape)


X, y = clean_data(
    train_df,
    has_target=True
)

X_test = clean_data(
    test_df,
    has_target=False
)


# ============================================================
# SAVE STUDENT IDs
# ============================================================

student_ids = X_test["student_id"].copy()


X = X.drop(columns=["student_id"])
X_test = X_test.drop(columns=["student_id"])


categorical_features = [
    "event_type",
    "club_member",
    "event_day"
]

numeric_features = [
    "registration_days_before",
    "previous_events_registered",
    "previous_events_attended",
    "travel_distance_km",
    "event_hour",
    "previous_attendance_rate"
]


print("\nCategorical features:")
print(categorical_features)

print("\nNumeric features:")
print(numeric_features)

print("\nTraining column data types:")
print(X.dtypes)


numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    )
])


categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    )
])

preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numeric_features
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_features
    )
])


model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=3,
            random_state=42
        )
    )
])



X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining model...")

model.fit(
    X_train,
    y_train
)

print("Training complete.")

val_predictions = model.predict(X_val)

val_probabilities = model.predict_proba(
    X_val
)[:, 1]


precision = precision_score(
    y_val,
    val_predictions
)

recall = recall_score(
    y_val,
    val_predictions
)

f1 = f1_score(
    y_val,
    val_predictions
)

roc_auc = roc_auc_score(
    y_val,
    val_probabilities
)

cm = confusion_matrix(
    y_val,
    val_predictions
)


print("\n====================================")
print("MODEL EVALUATION")
print("====================================")

print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nTraining final model on full dataset...")

model.fit(
    X,
    y
)

print("Final model training complete.")


print("\nGenerating test predictions...")

probabilities = model.predict_proba(
    X_test
)[:, 1]

predictions = (
    probabilities >= 0.5
).astype(int)

results = test_df.copy()

results["attendance_probability"] = probabilities

results["attendance_probability_pct"] = (
    probabilities * 100
).round(1)

results["predicted_attendance"] = predictions

results["prediction_label"] = np.where(
    predictions == 1,
    "Likely to Attend",
    "Unlikely to Attend"
)

results.to_excel(
    OUTPUT_FILE,
    index=False
)


print("\n====================================")
print("ATTENDANCE PREDICTIONS")
print("====================================")

print(
    results[
        [
            "student_id",
            "attendance_probability_pct",
            "prediction_label"
        ]
    ].to_string(index=False)
)


print("\n====================================")
print("DONE")
print("====================================")

print(f"Predictions saved to: {OUTPUT_FILE}")
