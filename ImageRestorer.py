import dippykit as dip
import numpy as np
from ImageFileHandler import ImageFileHandler
from ImageDegrader import ImageDegrader
import os
from sklearn import preprocessing
import cv2

class ImageRestorer:

    def __init__(self):
        pass

    def restore(self, image):
        return image

    def fast_multiplicative_restore(self, degraded_image, original_image, h_param=15):
        int_image = dip.float_to_im(degraded_image)
        psnr_max = None
        best_denoise = None
        for i in range(1, 25):
            cur_denoised = dip.im_to_float(cv2.fastNlMeansDenoising(int_image, h=i, searchWindowSize=31))
            cur_psnr = dip.PSNR(original_image, cur_denoised)
            if psnr_max is None or cur_psnr > psnr_max:
                best_denoise = cur_denoised
                psnr_max = cur_psnr
        return best_denoise

    def slow_multiplicative_restore(self, degraded_image):
        h = 15
        s_window_size = 21
        t_window_size = 7

    def calc_d_squared(self, im, p, q, f):
        scaling_factor = 1 / np.square(2 * f + 1)
        p_min = np.max([p - f, 0])
        q_min = np.max([q - f, 0])
        p_max = np.min([p + f, ])
        im_p_vec =

    def calc_weights(self, d_squared, sigma_squared, h):
        max_val = np.max([d_squared - 2*sigma_squared, 0.0])
        return np.exp(-max_val / np.square(h))


    def _test_restore_mode(self, file, deg_type):
        fh = ImageFileHandler()
        im_degrader = ImageDegrader()
        im = fh.open_image_file_as_matrix(file)
        degraded_im = im_degrader.degrade(im, degradation_type=deg_type)
        restored_im = self.fast_multiplicative_restore(degraded_im, im)
        dip.figure()
        dip.subplot(131)
        dip.imshow(im, cmap="gray")
        dip.subplot(132)
        dip.imshow(degraded_im, cmap="gray")
        dip.xlabel("PSNR: {0:.2f}".format(dip.PSNR(im, degraded_im)))
        dip.subplot(133)
        dip.imshow(restored_im, cmap="gray")
        dip.xlabel("PSNR: {0:.2f}".format(dip.PSNR(im, restored_im)))
        dip.show()

if __name__ == "__main__":
    file = os.path.join(os.getcwd(), "test_images", "cameraman.jpeg")
    ir = ImageRestorer()
    ir._test_restore_mode(file, deg_type="multiplicative")
