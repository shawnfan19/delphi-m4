import numpy as np

from delphi.data.aou import MultimodalAOUReader as Reader


def test_train_complements_val():
    """train fold = every participant NOT in the held-out 'val' stride."""
    allp = Reader.participants("all")
    train = Reader.participants("train")
    val = Reader.participants("val")

    # train and val are disjoint and together cover the whole cohort
    assert np.intersect1d(train, val).size == 0
    assert set(train) | set(val) == set(allp)

    # train is exactly the union of the non-'val' CV strides
    others = np.concatenate(
        [Reader.participants(f) for f in Reader.FOLDS if f != "val"]
    )
    assert set(train) == set(others)
