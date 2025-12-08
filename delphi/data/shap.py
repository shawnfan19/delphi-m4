import numpy as np
import shap


class ShapMasker(shap.maskers.Masker):  # type: ignore

    def __init__(self):
        pass

    def shape(self, s: tuple[np.ndarray, np.ndarray]):
        return (1, len(s[0]))

    def mask_shapes(self, s: tuple[np.ndarray, np.ndarray]):
        return [(len(s[0]),)]

    def __call__(self, mask, s: tuple[np.ndarray, np.ndarray]):
        mask = self._standardize_mask(mask, s)
        x, t = s
        x, t = x.copy(), t.copy()

        # NumPy advanced indexing returns copy, not view
        masked_x = x[~mask]
        masked_t = t[~mask]

        masked_x[masked_x > 3] = 0
        masked_t[masked_x > 3] = -1e4

        masked_x[masked_x == 2] = 3
        masked_x[masked_x == 3] = 2

        x[~mask] = masked_x
        t[~mask] = masked_t

        sort_idx = np.argsort(t)
        x = np.take_along_axis(x, sort_idx, axis=0)
        t = np.take_along_axis(t, sort_idx, axis=0)

        return ((x,), (t,))
