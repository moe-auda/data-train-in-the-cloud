from unittest.mock import patch
import shutil
import pytest
import pickle
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from taxifare.params import *

DATA_SIZE = "1k"
CHUNK_SIZE = 200

MIN_DATE='2009-01-01'
MAX_DATE='2015-01-01'

@patch("taxifare.params.DATA_SIZE", new=DATA_SIZE)
@patch("taxifare.params.CHUNK_SIZE", new=CHUNK_SIZE)
class TestMain():
    """Assert that code logic runs and outputs the correct type. Do not check model performance"""

    def test_route_preprocess(self, fixture_query_1k):

        from taxifare.interface.main import preprocess

        data_query_path = Path(LOCAL_DATA_PATH).joinpath("raw",f"query_{MIN_DATE}_{MAX_DATE}_{DATA_SIZE}.csv")
        data_processed_path = Path(LOCAL_DATA_PATH).joinpath("processed",f"processed_{MIN_DATE}_{MAX_DATE}_{DATA_SIZE}.csv")

        data_query_exists = data_query_path.is_file()
        data_processed_exists = data_processed_path.is_file()

        # SETUP
        if data_query_exists:
            shutil.copyfile(data_query_path, f'{data_query_path}_backup')
            data_query_path.unlink()
        if data_processed_exists:
            shutil.copyfile(data_processed_path, f'{data_processed_path}_backup')
            data_processed_path.unlink()

        # ACT
        # TODO: add try-except to be certain of reseting state if crash
        # Check route runs querying Big Query
        preprocess(min_date=MIN_DATE, max_date=MAX_DATE)

        # Load newly saved query data and test it against true fixture
        data_query = pd.read_csv(data_query_path, parse_dates=["pickup_datetime"])
        assert data_query.shape[1] == fixture_query_1k.shape[1], "Incorrect number of columns in your raw query CSV"
        assert data_query.shape[0] == fixture_query_1k.shape[0], "Incorrect number of rows in your raw query CSV. Did you append all chunks correcly ?"
        assert np.allclose(data_query[['fare_amount']].head(1), fixture_query_1k[['fare_amount']].head(1), atol=1e-3), "First row differs. Did you forgot to store headers in your preprocessed CSV ?"
        assert np.allclose(data_query[['fare_amount']].tail(1), fixture_query_1k[['fare_amount']].tail(1), atol=1e-3), "Last row differs somewhow"


        # Check again that route runs, this time loading local CSV cached
        preprocess(min_date=MIN_DATE, max_date=MAX_DATE)

        # RESET STATE
        data_query_path.unlink(missing_ok=True)
        data_processed_path.unlink(missing_ok=True)
        if data_query_exists:
            shutil.move(f'{data_query_path}_backup', data_query_path)
        if data_processed_exists:
            shutil.move(f'{data_processed_path}_backup', data_processed_path)

    @pytest.mark.parametrize('model_target', ['local' , 'gcs'])
    def test_route_train(self, fixture_processed_1k, model_target):
        """Test route train behave as expected, for various context of LOCAL or GCS model storage"""

        # 1) SETUP
        old_model_target = os.environ.get("MODEL_TARGET")
        os.environ.update(MODEL_TARGET=model_target)

        data_processed_path = Path(LOCAL_DATA_PATH).joinpath("processed",f"processed_{MIN_DATE}_{MAX_DATE}_{DATA_SIZE}.csv")
        data_processed_exists = data_processed_path.is_file()
        if data_processed_exists:
            shutil.copyfile(data_processed_path, f'{data_processed_path}_backup')
            data_processed_path.unlink()

        data_processed_fixture_path = "https://storage.googleapis.com/datascience-mlops/taxi-fare-ny/solutions/data_processed_fixture_2009-01-01_2015-01-01_1k.csv"
        os.system(f"curl {data_processed_fixture_path} > {data_processed_path}")

        # 2) ACT
        from taxifare.interface.main import train

        # Train it from Big Query
        train(min_date=MIN_DATE, max_date=MAX_DATE, learning_rate=0.01, patience=0)

        assert data_processed_path.is_file(), "you should store processed CSV as cached"
        data_processed = pd.read_csv(data_processed_path, header=None, dtype=DTYPES_PROCESSED)
        assert data_processed.shape[1] == fixture_processed_1k.shape[1], "Incorrect number of columns in your processed CSV. There should be 66 (65 features data_processed + 1 target)"
        assert data_processed.shape[0] == fixture_processed_1k.shape[0], "Incorrect number of rows in your processed CSV. Did you append all chunks correcly ?"
        assert np.allclose(data_processed.head(1), fixture_processed_1k.head(1), atol=1e-3), "First row differs. Did you store headers ('1', '2', ...'65') in your processed CSV by mistake?"
        assert np.allclose(data_processed, fixture_processed_1k, atol=1e-3), "One of your data processed value is somehow incorrect!"

        # Train it from local CSV
        train(learning_rate=0.01, patience=0)

        # RESET STATE
        os.environ.update(MODEL_TARGET=old_model_target)
        data_processed_path.unlink(missing_ok=True)
        if data_processed_exists:
            shutil.move(f'{data_processed_path}_backup', data_processed_path)

    @patch("taxifare.params.MODEL_TARGET", new='local')
    def test_route_evaluate(self):
        from taxifare.interface.main import evaluate

        mae = evaluate(min_date=MIN_DATE, max_date=MAX_DATE)

        assert isinstance(mae, float), "calling evaluate() should return the MAE as a float"

    @patch("taxifare.params.MODEL_TARGET", new='local')
    def test_route_pred(self):
        from taxifare.interface.main import pred

        y_pred = pred()
        pred_value = y_pred.flat[0].tolist()

        assert isinstance(pred_value, float), "calling pred() should return a float"
