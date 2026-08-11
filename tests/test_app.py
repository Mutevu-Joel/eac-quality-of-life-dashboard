from pathlib import Path

from streamlit.testing.v1 import AppTest


PROJECT_DIR = Path(__file__).resolve().parents[1]


def test_dashboard_renders_without_exceptions():
    app = AppTest.from_file(PROJECT_DIR / "app.py", default_timeout=120)
    app.run()
    assert not app.exception
    assert len(app.metric) >= 4
    assert len(app.tabs) == 4
    assert len(app.multiselect) >= 1
    assert len(app.slider) >= 1
    assert any(widget.label == "Map indicator" for widget in app.selectbox)
