from pathlib import Path


def test_promoted_mantis_base_is_encoder_only_mv_v3():
    from futures_foundation.finetune.classifiers.mantis import BASE_CKPT

    root = Path(__file__).resolve().parents[1]
    assert BASE_CKPT == "checkpoints/mantis_ssl_mv_v3.pt"
    encoder = root / BASE_CKPT
    assert encoder.is_file()
