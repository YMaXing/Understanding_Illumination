import sys
from pathlib import Path

import pytest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config as K  # noqa: E402
from core import io as IO  # noqa: E402

DEV = ["taiwei_cui_070", "taiwei_cui_075", "taiwei_cui_076", "taiwei_cui_079", "taiwei_cui_080",
       "taiwei_cui_082", "taiwei_cui_083", "taiwei_cui_084", "taiwei_cui_088", "taiwei_cui_090"]


@pytest.fixture(scope="session")
def cfg():
    return K.load_config()


def need_dataset(cfg, stems=DEV):
    """Skip (clear message) when the local dataset is not configured or lacks an image."""
    try:
        for s in stems:
            K.image_path(cfg, s)
    except FileNotFoundError as e:
        pytest.skip(f"local dataset missing ({e}); set dataset.val_set in metrics/config.local.yaml")


def need_migration_sources(cfg):
    """Skip (clear message) when the old pair files migrate_v0.py verify reads are absent (they are not on the
    published metric-pair-annotation branch; verify is maintainer-only)."""
    miss = [s["source"] for s in cfg["sets"].values() if s.get("source") and not K.repo_path(s["source"]).exists()]
    if miss:
        pytest.skip(f"migration source files absent ({len(miss)}, e.g. {miss[0]}); migrate_v0.py verify is maintainer-only")


@pytest.fixture(scope="session")
def dev_images(cfg):
    need_dataset(cfg)
    return {s: IO.load_image(cfg, s) for s in DEV}
