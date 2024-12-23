from fastapi.testclient import TestClient
from unittest.mock import MagicMock
import sys
import mlflow

sys.path.append("../")


from app import app, minio_client


def test_upload_dataset_with_mock(tmp_path):
    """
    Тест для проверки загрузки датасета с мокированием MinIO.
    """
    client = TestClient(app)
    minio_client.fput_object = MagicMock(return_value=None)

    test_file = tmp_path / "test_dataset.csv"
    test_file.write_text("col1,col2\n1,2\n3,4")

    with open(test_file, "rb") as f:
        response = client.post(
            "/upload_dataset", files={"file": ("test_dataset.csv", f, "text/csv")}
        )

    assert response.status_code == 200
    assert response.json() == {"message": "Successfully uploaded test_dataset.csv"}

    minio_client.fput_object.assert_called_once_with(
        "datasets", "test_dataset.csv", "test_dataset.csv"
    )
