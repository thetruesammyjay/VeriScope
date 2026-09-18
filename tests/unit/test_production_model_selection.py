from pathlib import Path

from apps.api.api import dependencies
from apps.api.core.config import Settings


class _FakeTransformerPredictor:
    def predict(self, text: str):
        del text
        return None


def test_build_inference_service_uses_transformer_when_selected(monkeypatch):
    expected_path = Path("models/transformer/release")
    loaded_paths: list[Path] = []
    fake_predictor = _FakeTransformerPredictor()
    monkeypatch.setattr(
        dependencies,
        "load_transformer_predictor",
        lambda path: loaded_paths.append(path) or fake_predictor,
    )
    settings = Settings(
        _env_file=None,
        production_model="transformer",
        transformer_model_path=expected_path,
    )

    service = dependencies.build_inference_service(settings)

    assert service.predictor is fake_predictor
    assert loaded_paths == [expected_path]


def test_build_inference_service_reports_unavailable_transformer(monkeypatch):
    monkeypatch.setattr(
        dependencies,
        "load_transformer_predictor",
        lambda path: (_ for _ in ()).throw(FileNotFoundError(path)),
    )
    settings = Settings(_env_file=None, production_model="transformer")

    assert dependencies.build_inference_service(settings).available is False
