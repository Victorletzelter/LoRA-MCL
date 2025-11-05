"""
Basic import and initialization tests for peft_mcl package.
"""

import pytest


def test_imports():
    """Test that all main components can be imported."""
    try:
        from peft_mcl import (
            MCLConfig,
            MCLForwardMixin,
            MCLModelWrapper,
            MCLModelOutput,
            get_peft_mcl,
            MCLTrainer,
            patch_peft_for_mcl,
        )
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import peft_mcl components: {e}")


def test_version():
    """Test that version is defined."""
    import peft_mcl
    assert hasattr(peft_mcl, "__version__")
    assert isinstance(peft_mcl.__version__, str)


def test_all_exports():
    """Test that __all__ is properly defined."""
    import peft_mcl
    assert hasattr(peft_mcl, "__all__")
    assert isinstance(peft_mcl.__all__, list)
    assert len(peft_mcl.__all__) > 0


def test_mcl_config_exists():
    """Test that MCLConfig class exists and can be imported."""
    from peft_mcl import MCLConfig
    assert MCLConfig is not None


def test_patch_function_callable():
    """Test that patch_peft_for_mcl is callable."""
    from peft_mcl import patch_peft_for_mcl
    assert callable(patch_peft_for_mcl)


def test_get_peft_mcl_callable():
    """Test that get_peft_mcl is callable."""
    from peft_mcl import get_peft_mcl
    assert callable(get_peft_mcl)


def test_trainer_class_exists():
    """Test that MCLTrainer class exists."""
    from peft_mcl import MCLTrainer
    assert MCLTrainer is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

