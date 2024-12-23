from fastapi.testclient import TestClient
import sys

sys.path.append("../")


from app import app, models_manager
from unittest.mock import MagicMock

client = TestClient(app)


def test_get_available_methods():
    """
    Тестирует endpoint /get_available_methods.
    """
    available_methods = ["method1", "method2", "method3"]
    models_manager.get_available_classes = MagicMock(return_value=available_methods)

    response = client.get("/get_available_methods")

    assert response.status_code == 200

    assert response.json() == {"message": available_methods}

    models_manager.get_available_classes.assert_called_once()
