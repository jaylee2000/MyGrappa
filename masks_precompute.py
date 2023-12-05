import numpy as np

masks = {
    1: np.array([[[False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False]],

       [[False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False]],

       [[False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False]],

       [[False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False]],

       [[False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False]]]),
    2: np.array([[[False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False]],

       [[False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False]],

       [[False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False]],

       [[False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False]],

       [[False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False]]]),
    6: np.array([[[False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False]],

       [[False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False]],

       [[False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False]],

       [[False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False]],

       [[False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False]]]),
    7: np.array([[[False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False]],

       [[False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False]],

       [[False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False]],

       [[False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False]],

       [[False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False]]]),
    13: np.array([[[False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False]],

       [[False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False]],

       [[False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False]],

       [[False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False]],

       [[False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False]]]),
    14: np.array([[[False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False]],

       [[False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False]],

       [[False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False]],

       [[False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False]],

       [[False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False]]]),
    15: np.array([[[False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False]],

       [[False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False]],

       [[False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False]],

       [[False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False]],

       [[False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False]]]),
    16: np.array([[[False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False]],

       [[False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False]],

       [[False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False]],

       [[False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False]],

       [[False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False]]]),
    17: np.array([[[False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False]],

       [[False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False]],

       [[False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False]],

       [[False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False]],

       [[False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False]]]),
    18: np.array([[[False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False]],

       [[False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False]],

       [[False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False]],

       [[False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False]],

       [[False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False],
        [ True,  True,  True,  True],
        [False, False, False, False]]])
}