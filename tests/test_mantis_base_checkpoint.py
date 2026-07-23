from pathlib import Path


def test_promoted_mantis_base_is_mv_and_has_matched_readout():
    from futures_foundation.finetune.classifiers.mantis import BASE_CKPT

    root = Path(__file__).resolve().parents[1]
    assert BASE_CKPT == "checkpoints/mantis_ssl_mv.pt"
    encoder = root / BASE_CKPT
    readout = Path(str(encoder) + ".readout.pt")
    assert encoder.is_file()
    assert readout.is_file()
