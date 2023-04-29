import os
import numpy as np
import pytest
from google.cloud import bigquery

from tests.test_base import TestBase
from taxifare.params import *

class TestCloudData(TestBase):
    def test_big_query_dataset_variable_exists(self):
        """
        Verify the BQ_DATASET variable is set
        """
        dataset = BQ_DATASET

        assert dataset is not None


    def test_cloud_data_create_dataset(self):
        """
        verify that the bq dataset is created and the Makefile variable correct
        """
        dataset = BQ_DATASET
        if dataset is None:
            raise ValueError("The BQ_DATASET environment variable is not set")
        client = bigquery.Client(project=GCP_PROJECT)
        datasets = [dataset.dataset_id for dataset in client.list_datasets()]

        assert dataset in datasets, f"Dataset {dataset} does not exist on the active GCP project"


    def test_cloud_data_create_table(self):
        """
        verify that the bq dataset tables are created and the Makefile variables correct
        """
        expected_tables = ["processed_1k", "processed_200k", "processed_all"]
        dataset = BQ_DATASET
        if dataset is None:
            raise ValueError("The BQ_DATASET environment variable is not set")
        client = bigquery.Client(project=GCP_PROJECT)
        tables = [table.table_id for table in client.list_tables(dataset)]
        for table in expected_tables:
            assert table in tables, f"Table {table} is missing from the {dataset} dataset"
