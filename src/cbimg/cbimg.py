""" File:
		cb-img.py

	Implements color balancing algorithms.
"""
import numpy as np
from scipy.stats.mstats import mquantiles
import concurrent.futures

class CBImg:
    """
    Implemented Public Methods:
    - grayWorld
	- simplestColorBalance
	- robustAWB

    Every public method expects an RGB image and returns an RGB image.
    """
    __sRGBtoXYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                            [0.2126729, 0.7151522, 0.0721750],
                            [0.0193339, 0.1191920, 0.9503041]])


    def __init__(self):
        pass

    def __cb_reshape(self, img):
        """
        Takes an M x N x 3 RGB image and returns a 3 x (M*N) matrix,
        where each column is a RGB pixel.
        """
        return np.transpose(img, (2, 0, 1)).reshape((3, np.size(img)//3))

    def __cb_unshape(self, matrix, height, width):
        """
        Takes a 3 x (M*N) matrix and returns a RGB M x N x 3 matrix.
        """
        return np.transpose(matrix.reshape((3, height, width)), (1, 2, 0))

    def __XYZ_to_xy(self, xyz):
        """
		Converts CIE XYZ to xy chromaticity.
        """
        X = xyz[0:1][0]
        Y = xyz[1:2][0]
        s = np.array([sum(xyz)])
        return np.array([X/s, Y/s])

    def __xy_to_XYZ(self, xy, Y):
        """
        Converts xyY chromaticity to CIE XYZ.
		"""
        x = xy[0]
        y = xy[1]
        return np.array([[(Y/y * x), Y, Y/y * (1 - x - y)]], dtype='float64')

    def __make_col(self, x):
        s = x.shape
        if len(s) == 2 and s[0] < s[1]:
            x = x.transpose()
        return x

    def __cb_CAT(self, xyz_est, xyz_target, cat_type):
        """
        Chromatic Adaptation Transform.
        """
        xyz_est    = self.__make_col(xyz_est)
        xyz_target = self.__make_col(xyz_target)
        xfm        = None

        if cat_type == "vonKries":
            xfm = np.array([[0.40024,  0.7076, -0.08081],
                            [-0.2263, 1.16532,   0.0457],
                            [    0.0,     0.0,  0.91822]])
        elif cat_type == "bradford":
            xfm = np.array([[ 0.8951,  0.2664, -0.1614],
                            [-0.7502,  1.7135,  0.0367],
                            [ 0.0389, -0.0685,  1.0296]])
        elif cat_type == "sharp":
            xfm = np.array([[ 1.2694, -0.0988, -0.1706],
                            [-0.8364,  1.8006,  0.0357],
                            [ 0.0297, -0.0315,  1.0018]])
        elif cat_type == "cmccat2000":
            xfm = np.array([[ 0.7982, 0.3389, -0.1371],
                            [-0.5918, 1.5512,  0.0406],
                            [ 0.0008, 0.239,   0.9753]])
        elif cat_type == "cat02":
            xfm = np.array([[ 0.7328, 0.4296, -0.1624],
                            [-0.7036, 1.6975,  0.0061],
                            [ 0.0030, 0.0136,  0.9834]])
        else:
            raise ValueError("invalid type for cat_type")

        ### xfm^(-1) * diagflat(gain) * xfm
        gain              = np.dot(xfm, xyz_target) / np.dot(xfm, xyz_est)
        solution, _, _, _ = np.linalg.lstsq(xfm, np.diagflat(gain), rcond=None)
        solution          = np.dot(solution, xfm)
        # ###
        retsolution, _, _, _ = np.linalg.lstsq(self.__sRGBtoXYZ, solution, rcond=None)
        return np.dot(retsolution, self.__sRGBtoXYZ)

	#############################################
	#############################################

	#############################################
	# Public Methods                            #
    #############################################

    def grayWorld(self, img, *, cat_type="vonKries", max_iter=1):
        """!
        Color balancing using the Gray World assumption and Chromatic Adaptation
        Transform (CAT).

        \param img  RGB image
        \param cat_type  string with the CAT type.
			             Exactly one of:
			              * vonKries
			              * bradford
			              * sharp
			              * cmccat2000
			              * cat02
		\param max_iter maximum number of iterations

        \return The image @c img with its colors balanced
		"""
        assert type(img) is np.ndarray, 'img is not numpy.ndarray'
        assert len(img.shape) == 3 and img.shape[2] == 3, 'img must be in RGB color scheme'

        img_rgb = img/255
        height, width, _ = img.shape

        xyz_D65 = np.array([[95.04], [100.], [108.88]])
        b       = .001 # convergence limit

        img_orig = self.__cb_reshape(img_rgb) * 255
        graydiff = []
        for i in range(max_iter):
            rgb_est = np.array([np.mean(img_orig, axis=1)])
            rgb_est = rgb_est.transpose()

            graydiff.append(np.linalg.norm(np.array([rgb_est[0] - rgb_est[1],
                                                     rgb_est[0] - rgb_est[2],
                                                     rgb_est[1] - rgb_est[2]])))
            if graydiff[-1] < b: # Convergence
                break
            elif i >= 1 and abs(graydiff[-2] - graydiff[-1]) < 10e-6:
                break

            xy_est  = self.__XYZ_to_xy(np.dot(self.__sRGBtoXYZ, rgb_est))
            xyz_est = self.__xy_to_XYZ(xy_est, 100)                # normalize Y to 100 for D-65 luminance comparable
            img_rgb = np.dot(self.__cb_CAT(xyz_est, xyz_D65, cat_type), img_orig)

        out = self.__cb_unshape(img_rgb, height, width)
        np.clip(out, 0, 255, out=out)
        return np.uint8(out)

    def simplestColorBalance(self, img, *, sat_level=0.01):
        """!
        Color balancing through histogram normalization.

        \param img  RGB image
        \param sat_level  controls the percentage of  pixels clipped to
                          black and white

        \return the image with its colors balanced
		"""
        assert type(img) is np.ndarray, 'img is not numpy.ndarray'
        assert len(img.shape) == 3 and img.shape[2] == 3, 'img must be in RGB color scheme'

        height, width, _ = img.shape

        q        = np.array([sat_level/2.0, 1 - sat_level/2.0])
        img_orig = self.__cb_reshape(img/255) * 255
        img_rgb  = np.zeros(img_orig.shape)

        def __closure(c):
            low, high  = mquantiles(img_orig[c], q, alphap=0.5, betap=0.5)
            # Saturate appropriate points in distribution
            img_rgb[c] = np.where(img_orig[c] < low, low,
                                  (np.where(img_orig[c] > high, high, img_orig[c])))
            bottom     = np.amin(img_rgb[c])
            top        = np.amax(img_rgb[c])
            d          = top - bottom
            img_rgb[c] = (img_rgb[c] - bottom) * 255 / (d if d != 0 else 1)

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as e:
            for c in [0, 1, 2]:
                e.submit(__closure, c)

        out = self.__cb_unshape(img_rgb, height, width)
        np.clip(out, 0, 255, out=out)
        return np.uint8(out)

    def robustAWB(self, img, *, option="CAT", cat_type="vonKries", thresh=0.3, max_iter=1):
        """!
        Color balancing through 'robust auto white' estimating gray pixels based
        on its deviation from YUV space. Then applying a iterative correction by
        using Chromatic Adaptation Transform or directly adjusting the  channels
        R and B.

		\param img  an RGB image
        \param option  the correction method (RBgain or CAT)
		\param cat_type  the CAT type used if the option argument is CAT
			              * vonKries
			              * bradford
			              * sharp
			              * cmccat2000
			              * cat02
		\param thresh  the deviation limit from gray to consider
		\param max_iter  the maximum number of iterations

        \return the RGB image with its colors balanced
		"""
        assert type(img) is np.ndarray, 'img is not numpy.ndarray'
        assert len(img.shape) == 3 and img.shape[2] == 3, 'img must be in RGB color scheme'

        img_rgb = img/255
        height, width, _ = img.shape

        xyz_D65 = np.array([[95.04], [100.], [108.88]])
        u = .01   # gain step
        a = .8    # double step limit
        b = .001  # convergence limit

        # RGB to YUV
        xfm = np.array([[ 0.299,  0.587,  0.114],
                        [-0.299, -0.587,  0.886],
                        [ 0.701, -0.587, -0.114]])

        img_orig = self.__cb_reshape(img_rgb) * 255
        img_rgb = img_orig.copy()
        gain = np.array([1.0, 1.0, 1.0])

        U_avg = []
        V_avg = []
        gray_total = np.array([])

        for i in range(max_iter):
            # to YUV
            img = np.dot(xfm, img_rgb)
            # Find gray chromaticity (|U|+|V|)/Y
            with np.errstate(divide='ignore', invalid='ignore'):
                F = np.array((abs(img[1]) + abs(img[2])) / img[0])#[0]

            gray_total = np.append(gray_total, sum(F<thresh))
            if gray_total[-1] == 0: # Valid gray pixels not found
                break

            grays = img[:, F<thresh]
            U_bar = np.mean(grays[1])
            V_bar = np.mean(grays[2])
            U_avg.append(U_bar)
            V_avg.append(V_bar)

            if option == "CAT" and cat_type:
                if max(abs(np.array([U_bar, V_bar]))) < b: # converged
                    break
                elif i >= 2 and np.linalg.norm(np.array([U_avg[-1] - U_avg[-2], V_avg[-1] - V_avg[-2]])) < 10e-6:
                    break

                # Convert gray average from YUV to RGB
                rgb_est, _, _, _ = np.linalg.lstsq(xfm, np.array([[100.], [U_bar], [V_bar]]), rcond=None)
                # xy chromaticity
                xy_est = self.__XYZ_to_xy(np.dot(self.__sRGBtoXYZ, rgb_est))
                # Normalize Y to 100 to be luminance D65 comparable
                xyz_est = self.__xy_to_XYZ(xy_est, 100.)

                img_rgb = np.dot(self.__cb_CAT(xyz_est, xyz_D65, cat_type), img_rgb)
            elif option == "RBgain":
                if abs(U_bar) > abs(V_bar): # U > V: blue needs adjust
                    err = U_bar
                    chnl = 2 # blue
                else:
                    err = V_bar
                    chnl = 0 # red

                if abs(err) >= a:
                    delta = 2 * np.sign(err) * u
                elif abs(err) < b:  # converged
                    delta = 0.
                    break
                else:
                    delta = err * u

                gain[chnl] = gain[chnl] - delta
                img_rgb = np.dot(np.diag(gain), img_orig)
            else:
                if cat_type == None:
                    raise ValueError("cat_type must be provided")
                else:
                    raise ValueError("invalid argument for 'option'")

        out = self.__cb_unshape(img_rgb, height, width)
        np.clip(out, 0, 255, out=out)
        return np.uint8(out)
