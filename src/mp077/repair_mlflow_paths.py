import hashlib
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "mlflow.db"

OLD_PREFIX = "/Users/aadya/Downloads/G067-MP077/"

EXPECTED_DB_HASH = (
    "3f398dc83bcf79300e4a11a241b0b418c67474a4609c6d028664e72f46bb9fcc"
)

PATH_COLUMNS = [
    ("runs", "artifact_uri"),
    ("logged_models", "artifact_location"),
    ("experiments", "artifact_location"),
    ("model_versions", "storage_location"),
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repair():

    print("===== PRE-REPAIR SAFETY =====")

    assert DB.exists(), "MLflow database not found"

    con = sqlite3.connect(DB)

    try:

        con.execute("BEGIN IMMEDIATE")

        runs_before = con.execute(
            "SELECT * FROM runs ORDER BY run_uuid"
        ).fetchall()

        versions_before = con.execute(
            """
            SELECT name, version, run_id, source
            FROM model_versions
            ORDER BY name, version
            """
        ).fetchall()

        aliases_before = con.execute(
            """
            SELECT name, alias, version
            FROM registered_model_aliases
            ORDER BY name, alias
            """
        ).fetchall()

        changes = []

        for table, column in PATH_COLUMNS:

            rows = con.execute(
                f"""
                SELECT DISTINCT {column}
                FROM {table}
                WHERE {column} LIKE ?
                """,
                (OLD_PREFIX + "%",)
            ).fetchall()

            for (old_path,) in rows:

                new_path = old_path[len(OLD_PREFIX):]

                assert new_path.startswith("mlruns/")

                if table in (
                    "logged_models",
                    "model_versions"
                ):

                    assert (
                        ROOT / new_path / "MLmodel"
                    ).is_file(), "MLmodel file missing"

                    assert (
                        ROOT / new_path / "model.pkl"
                    ).is_file(), "Model artifact missing"

                changes.append(
                    (table, column, old_path, new_path)
                )

        if not changes:

            con.rollback()

            print("No matching legacy paths found.")
            print("No database changes performed.")

            return

        assert sha256(DB) == EXPECTED_DB_HASH, (
            "Original database hash mismatch."
        )

        assert len(runs_before) == 7
        assert len(changes) == 17

        assert versions_before == [
            (
                "mp077-model",
                1,
                "22ac88d2bb0b43e18302e81f02348a03",
                "models:/m-f4a06f82005e4a9b9b2c2157c838c397"
            )
        ]

        assert aliases_before == [
            ("mp077-model", "champion", 1)
        ]

        print("Original database verified")

        print("\n===== APPLYING PATH REPAIR =====")

        total_updated = 0

        for table, column, old_path, new_path in changes:

            cursor = con.execute(
                f"""
                UPDATE {table}
                SET {column} = ?
                WHERE {column} = ?
                """,
                (new_path, old_path)
            )

            total_updated += cursor.rowcount

            print(f"\n{table}.{column}")
            print("Old:", old_path)
            print("New:", new_path)

        assert total_updated == 17

        # Verify existing training history
        runs_after = con.execute(
            "SELECT * FROM runs ORDER BY run_uuid"
        ).fetchall()

        run_columns = [
            row[1]
            for row in con.execute(
                "PRAGMA table_info(runs)"
            )
        ]

        artifact_index = run_columns.index(
            "artifact_uri"
        )

        def without_artifact(row):

            return tuple(
                value
                for i, value in enumerate(row)
                if i != artifact_index
            )

        assert [
            without_artifact(row)
            for row in runs_before
        ] == [
            without_artifact(row)
            for row in runs_after
        ]

        # Verify registered model versions
        versions_after = con.execute(
            """
            SELECT name, version, run_id, source
            FROM model_versions
            ORDER BY name, version
            """
        ).fetchall()

        # Verify champion alias
        aliases_after = con.execute(
            """
            SELECT name, alias, version
            FROM registered_model_aliases
            ORDER BY name, alias
            """
        ).fetchall()

        assert versions_before == versions_after
        assert aliases_before == aliases_after

        # Verify SQLite database integrity
        integrity = con.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]

        assert integrity == "ok"

        # Permanently save verified changes
        con.commit()

        print("\n===== REPAIR SUMMARY =====")

        print("Updated references:", total_updated)
        print("Database integrity:", integrity)
        print("Training runs preserved:", len(runs_after))
        print("Registered versions preserved:", True)
        print("Champion alias preserved:", True)
        print("Champion version: 1")
        print("Artifact path repair completed")
        print("No model retraining performed")

    except Exception:

        con.rollback()

        print("\nRepair failed.")
        print("Transaction rolled back.")

        raise

    finally:

        con.close()


if __name__ == "__main__":
    repair()