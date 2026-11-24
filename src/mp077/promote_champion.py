import sqlite3

import mlflow
import numpy as np

from sklearn.metrics import mean_absolute_error

from mp077.run_stage4 import (
    ROOT,
    DB,
    REGISTERED_NAME,
    TRAIN_PATH,
    LATER_PATH,
    FEATURES,
    MIN_IMPROVEMENT,
    sha256,
    validate_data,
    prepare_split,
)


V1_RUN_ID = "22ac88d2bb0b43e18302e81f02348a03"
V2_RUN_ID = "df8bc1f1d3b3421b9dbbbc7311460f45"

EXPECTED_V1_HASH = (
    "21c2b5946fd6f5dca6d50a2e3fb42cd4495e3df69c6c6e2ada4bc85d13d50839"
)


def main():

    mlflow.set_tracking_uri(DB)
    mlflow.set_registry_uri(DB)

    client = mlflow.MlflowClient()

    model_path = ROOT / "models/model.joblib"

    assert sha256(model_path) == EXPECTED_V1_HASH

    print("===== PRE-PROMOTION VERIFICATION =====")

    versions = client.search_model_versions(
        "name = 'mp077-model'"
    )

    assert {str(v.version) for v in versions} == {
        "1", "2"
    }

    current = client.get_model_version_by_alias(
        REGISTERED_NAME,
        "champion"
    )

    assert str(current.version) in {"1", "2"}

    print("Current champion:", current.version)

    v1_info = client.get_model_version(
        REGISTERED_NAME,
        "1"
    )

    v2_info = client.get_model_version(
        REGISTERED_NAME,
        "2"
    )

    assert v1_info.run_id == V1_RUN_ID
    assert v2_info.run_id == V2_RUN_ID

    assert v1_info.status == "READY"
    assert v2_info.status == "READY"

    run = client.get_run(V2_RUN_ID)

    assert run.info.status == "FINISHED"

    print("\n===== TRACKING METADATA =====")

    assert run.data.tags["data_sha256"] == sha256(
        TRAIN_PATH
    )

    assert run.data.tags["later_data_sha256"] == sha256(
        LATER_PATH
    )

    assert run.data.tags["parent_model_version"] == "1"

    assert run.data.params["combined_train_rows"] == "1210"
    assert run.data.params["check_rows"] == "90"

    print("Dataset fingerprints: PASS")
    print("Training parameters: PASS")
    print("V2 run status: FINISHED")
    print("Registered V2 status: READY")

    print("\n===== LOAD REGISTERED MODELS =====")

    v1 = mlflow.sklearn.load_model(
        f"models:/{REGISTERED_NAME}/1"
    )

    v2 = mlflow.sklearn.load_model(
        f"models:/{REGISTERED_NAME}/2"
    )

    assert list(v1.feature_names_in_) == FEATURES
    assert list(v2.feature_names_in_) == FEATURES

    train, later = validate_data()

    _, _, X_check, y_check = prepare_split(
        train,
        later
    )

    p1 = v1.predict(X_check)
    p2 = v2.predict(X_check)

    assert np.isfinite(p1).all()
    assert np.isfinite(p2).all()

    mae_v1 = mean_absolute_error(y_check, p1)
    mae_v2 = mean_absolute_error(y_check, p2)

    assert mae_v1 > 0

    assert np.isclose(
        mae_v1,
        run.data.metrics["mae_v1_on_check"],
        rtol=1e-6
    )

    assert np.isclose(
        mae_v2,
        run.data.metrics["mae_v2_on_check"],
        rtol=1e-6
    )

    improvement = (
        (mae_v1 - mae_v2) / mae_v1
    ) * 100

    threshold = mae_v1 * (
        1 - MIN_IMPROVEMENT
    )

    eligible = mae_v2 <= threshold

    print("\n===== PROMOTION DECISION =====")

    print("V1 MAE:", mae_v1)
    print("V2 MAE:", mae_v2)
    print("Maximum allowed V2 MAE:", threshold)
    print("Improvement percentage:", improvement)
    print("Promotion eligible:", eligible)

    selected_version = "2" if eligible else "1"

    previous_version = str(current.version)

    try:

        if previous_version != selected_version:

            client.set_registered_model_alias(
                REGISTERED_NAME,
                "champion",
                selected_version
            )

        print("\n===== CHAMPION VERIFICATION =====")

        champion_info = client.get_model_version_by_alias(
            REGISTERED_NAME,
            "champion"
        )

        assert str(
            champion_info.version
        ) == selected_version

        champion_model = mlflow.sklearn.load_model(
            f"models:/{REGISTERED_NAME}@champion"
        )

        expected = p2 if eligible else p1

        actual = champion_model.predict(X_check)

        assert np.allclose(
            actual,
            expected,
            rtol=1e-9,
            atol=1e-8
        )

        print(
            "Maximum prediction difference:",
            np.max(np.abs(actual - expected))
        )

        print("Champion prediction verification: PASS")

        with sqlite3.connect(
            "file:mlflow.db?mode=ro",
            uri=True
        ) as con:

            integrity = con.execute(
                "PRAGMA integrity_check"
            ).fetchone()[0]

            total_runs = con.execute(
                "SELECT COUNT(*) FROM runs"
            ).fetchone()[0]

        assert integrity == "ok"
        assert total_runs == 8

        assert sha256(model_path) == EXPECTED_V1_HASH

    except Exception:

        client.set_registered_model_alias(
            REGISTERED_NAME,
            "champion",
            previous_version
        )

        print("Verification failed.")
        print("Previous champion alias restored.")

        raise

    print("\n===== FINAL PROMOTION SUMMARY =====")

    print("Previous champion:", previous_version)
    print("Selected champion:", selected_version)

    print(
        "Decision:",
        "promote" if eligible else "keep"
    )

    print("MAE improvement percentage:", improvement)
    print("Database integrity:", integrity)
    print("Total MLflow runs:", total_runs)
    print("Both registered versions preserved: True")
    print("Original exported V1 unchanged: True")
    print("No retraining performed")
    print("Champion selection completed successfully")


if __name__ == "__main__":
    main()