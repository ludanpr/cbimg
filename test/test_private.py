"""
File:
    test/test_private.py

$ pytest test_private.py
"""
import numpy as np
#import cbimg
#### REMOVE THIS FOR RELEASE
import sys
sys.path.insert(1, '../src/cbimg/')
import cbimg

def test_reshape_1():
    img = np.array([[[  0, 128, 255],
                     [255, 128,   0],
                     [  0,   0,   0],
                     [128, 128, 128],
                     [255, 255, 255]],
                    [[  0, 128, 255],
                     [255, 128,   0],
                     [  0,   0,   0],
                     [128, 128, 128],
                     [255, 255, 255]],
                    [[  0, 128, 255],
                     [255, 128,   0],
                     [  0,   0,   0],
                     [128, 128, 128],
                     [255, 255, 255]],
                    [[  0, 128, 255],
                     [255, 128,   0],
                     [  0,   0,   0],
                     [128, 128, 128],
                     [255, 255, 255]]])
    assert img.shape == (4, 5, 3)

    cb = cbimg.CBImg()
    r = cb._CBImg__cb_reshape(img)
    assert r.shape == (3, 20)
    assert np.array_equal(r, [[0, 255, 0, 128, 255, 0, 255, 0, 128, 255, 0, 255, 0, 128, 255, 0, 255, 0, 128, 255],
                              [128, 128, 0, 128, 255, 128, 128, 0, 128, 255, 128, 128, 0, 128, 255, 128, 128, 0, 128, 255],
                              [255, 0, 0, 128, 255, 255, 0, 0, 128, 255, 255, 0, 0, 128, 255, 255, 0, 0, 128, 255]])

def test_unshape_1():
    mat = np.array([[0, 255, 0, 128, 255, 0, 255, 0, 128, 255, 0, 255, 0, 128, 255, 0, 255, 0, 128, 255],
                    [128, 128, 0, 128, 255, 128, 128, 0, 128, 255, 128, 128, 0, 128, 255, 128, 128, 0, 128, 255],
                    [255, 0, 0, 128, 255, 255, 0, 0, 128, 255, 255, 0, 0, 128, 255, 255, 0, 0, 128, 255]])
    assert mat.shape == (3, 20)

    h, w = 4, 5   # Original img height and width
    cb = cbimg.CBImg()
    img = cb._CBImg__cb_unshape(mat, h, w)
    assert img.shape == (4, 5, 3)
    assert np.array_equal(img, [[[  0, 128, 255],
                                 [255, 128,   0],
                                 [  0,   0,   0],
                                 [128, 128, 128],
                                 [255, 255, 255]],
                                [[  0, 128, 255],
                                 [255, 128,   0],
                                 [  0,   0,   0],
                                 [128, 128, 128],
                                 [255, 255, 255]],
                                [[  0, 128, 255],
                                 [255, 128,   0],
                                 [  0,   0,   0],
                                 [128, 128, 128],
                                 [255, 255, 255]],
                                [[  0, 128, 255],
                                 [255, 128,   0],
                                 [  0,   0,   0],
                                 [128, 128, 128],
                                 [255, 255, 255]]])

def test_XYZ_to_xy_1():
    XYZ = np.array([[150],
                    [245],
                    [  0]])

    cb = cbimg.CBImg()
    xy = cb._CBImg__XYZ_to_xy(XYZ)
    assert abs(xy[0][0] - 0.379747) < 1e-6
    assert abs(xy[1][0] - 0.620253) < 1e-6

def test_xy_to_XYZ_1():
    xy = np.array([[[0.379747]],
                   [[0.620253]]])
    Y = 100

    cb = cbimg.CBImg()
    XYZ = cb._CBImg__xy_to_XYZ(xy, Y)
    print(XYZ)
    assert abs(XYZ[0, 0] - 61.224533) < 1e-6
    assert abs(XYZ[0, 1] - 100.0) < 1e-6
    assert abs(XYZ[0, 2] - 0.0) < 1e-6

def test_CAT_1():
    return True

test_xy_to_XYZ_1()
