import functools
import json
import os
import subprocess

import yaml
from google.cloud import bigquery, storage

from delphi.env import load_env_file

# Load <repo>/.env into os.environ if present (idempotent).
load_env_file()


_REQUIRED_VARS = (
    "GOOGLE_CLOUD_PROJECT",
    "WORKSPACE_CDR",
    "WORKSPACE_BUCKET",
    "WORKSPACE_TEMP_BUCKET",
    "CKPT_BUCKET",
    "DATA_BUCKET",
)


def _populate_env_from_wb() -> None:
    """Fall back to the Workbench CLI for any required env var not already set."""
    if all(k in os.environ for k in _REQUIRED_VARS):
        return

    workspace = json.loads(
        subprocess.run(
            ["wb", "workspace", "describe", "--format=json"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", workspace["googleProjectId"])

    resources = json.loads(
        subprocess.run(
            ["wb", "resource", "list", "--format=json"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )

    for r in resources:
        if r["resourceType"] == "GCS_BUCKET":
            print(f"Found bucket: id={r['id']}, bucketName={r['bucketName']}")
            if r["id"] == "data":
                os.environ.setdefault("DATA_BUCKET", r["bucketName"])
            elif r["id"] == "ckpt":
                os.environ.setdefault("CKPT_BUCKET", r["bucketName"])

            # Check temporary bucket first to avoid substring conflicts
            if "temporary-workspace-bucket" in r["id"]:
                os.environ.setdefault(
                    "WORKSPACE_TEMP_BUCKET", f"gs://{r['bucketName']}"
                )
            elif "workspace-bucket" in r["id"]:
                os.environ.setdefault("WORKSPACE_BUCKET", f"gs://{r['bucketName']}")

        elif r["resourceType"] in ("BQ_DATASET", "BIGQUERY_DATASET"):
            if not os.environ.get("WORKSPACE_CDR"):
                os.environ["WORKSPACE_CDR"] = f"{r['projectId']}.{r['datasetId']}"
                print(
                    f"Successfully set WORKSPACE_CDR to: {os.environ['WORKSPACE_CDR']}"
                )


_populate_env_from_wb()


PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"]
WORKSPACE_CDR = os.environ.get("WORKSPACE_CDR")
DATA_BUCKET = os.environ.get("DATA_BUCKET")


class Client:

    def __init__(self, dataset):
        self.dataset = dataset
        self.client = bigquery.Client()

    def tables(self):
        tables = self.client.list_tables(self.dataset)
        print(f"tables available in {self.dataset}:\n")
        # Loop through and print the name of each table
        for table in tables:
            print(f"- {table.table_id}")

    def list_columns(self, table_name):
        # get table metadata for free
        table = self.client.get_table(f"{self.dataset}.{table_name}")
        print(f"--- COLUMNS IN {table.table_id} ---")
        for field in table.schema:
            print(f"{field.name} ({field.field_type})")

    def list_rows(self, table_name):
        # get table metadata for free
        table = self.client.get_table(f"{self.dataset}.{table_name}")
        df_preview = self.client.list_rows(table, max_results=5).to_dataframe()
        return df_preview

    def run(self, query):
        # function to read data from BQ into py dataframe with using the Python client
        job_config = bigquery.QueryJobConfig(default_dataset=self.dataset)
        query_job = self.client.query(query, job_config=job_config)  # API request
        df = query_job.result().to_dataframe()
        return df

    def dry_run(self, query):
        # Configure the job to be a dry run
        job_config = bigquery.QueryJobConfig(
            dry_run=True, use_query_cache=False, default_dataset=self.dataset
        )
        # Send the query to BigQuery (this does NOT execute it or cost money)
        query_job = self.client.query(query, job_config=job_config)
        bytes_processed = query_job.total_bytes_processed

        if bytes_processed is not None:

            gb_processed = bytes_processed / (1024**3)
            tb_processed = bytes_processed / (1024**4)

            # cost based on Google's standard $6.25 per TB rate
            estimated_cost = tb_processed * 6.25

            print(f"data scanned: {gb_processed:.3f} GB")
            print(f"estimated Cost:     ${estimated_cost:.5f}")

            if estimated_cost > 1.00:
                print("⚠️ WARNING: This is an expensive query!")
        else:
            print("Could not estimate size. (Is the query syntax correct?)")

    def unique(self, table, column):
        q = f"""
        SELECT DISTINCT {column}
        FROM `{self.dataset}.{table}`
        """
        self.dry_run(q)
        return self.run(q)

    def value_counts(self, table, column):
        q = f"""
        SELECT
            {column},
            COUNT(*) AS frequency
        FROM `{self.dataset}.{table}`
        GROUP BY {column}
        ORDER BY frequency DESC
        """
        self.dry_run(q)
        return self.run(q)


@functools.cache
def _bucket():
    return storage.Client().bucket(DATA_BUCKET)


def upload_yaml(data, path: str) -> None:
    """Dump `data` to YAML and upload to gs://{DATA_BUCKET}/{path}."""
    _bucket().blob(path).upload_from_string(yaml.dump(data), content_type="text/yaml")
