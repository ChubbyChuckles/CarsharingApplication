import pytest

from src.main import main


def test_main(capsys: "pytest.CaptureFixture[str]") -> None:
    """Test for the Entry point for the project."""
    main()
    captured = capsys.readouterr()
    assert captured.out == "Hello from project-template!\n"
