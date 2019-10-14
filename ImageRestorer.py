import dippykit as dip
import numpy as np
from ImageFileHandler import ImageFileHandler
from ImageDegrader import ImageDegrader
import os


class ImageRestorer:

    def __init__(self):
        pass

    def restore(self, image):
        return image

    def AA_multiplicative_restore(self, image):

        lambda_param = 1000
        beta = 0.0000001
        dt = 1
        u_old = 0
        u_current = image
        epsilon = 0.001
        diff = 99
        iter = 0
        max_iter = 200
        while np.abs(diff) > epsilon and iter < max_iter:
            u_old = np.copy(u_current)
            div_func = self.AA_divergence_func(u_old, beta)
            h_func = self.AA_h_func(image, u_old)
            u_current = div_func + lambda_param * h_func + u_old
            diff = np.sum(np.subtract(u_current, u_old))
            iter += 1
        print("Ran {} iterations".format(iter))
        return self.normalize(u_current)

    def divergence(self, f):
        gradient = np.gradient(f)
        return gradient[0] + gradient[1]

    def normalize(self, v):
        norm = np.linalg.norm(v, ord=1)
        if norm == 0:
            norm = np.finfo(v.dtype).eps
        return v / norm

    def AA_h_func(self, f, u):
        return np.divide(np.subtract(f, u), (np.square(u) + 0.00001))

    def AA_divergence_func(self, u, beta):
        u_gradient = np.gradient(u)
        full_gradient = np.sqrt(u_gradient[0] ** 2 + u_gradient[1] ** 2)
        gradient_squared = np.square(np.abs(full_gradient))
        div = full_gradient / np.sqrt(gradient_squared + beta**2)
        return self.divergence(div)

    def _test_restore_mode(self, file, deg_type):
        fh = ImageFileHandler()
        im_degrader = ImageDegrader()
        im = fh.open_image_file_as_matrix(file)
        degraded_im = im_degrader.degrade(im, degradation_type=deg_type)
        restored_im = self.AA_multiplicative_restore(degraded_im)
        dip.figure()
        dip.subplot(131)
        dip.imshow(im, cmap="gray")
        dip.subplot(132)
        dip.imshow(degraded_im, cmap="gray")
        dip.xlabel("PSNR: {0:.2f}".format(dip.PSNR(im, degraded_im)))
        dip.subplot(133)
        dip.imshow(restored_im, cmap="gray")
        dip.xlabel("PSNR: {0:.2f}".format(dip.PSNR(im, restored_im, max_signal_value=np.max(restored_im))))
        dip.show()

if __name__ == "__main__":
    file = os.path.join(os.getcwd(), "test_images", "cameraman.jpeg")
    ir = ImageRestorer()
    ir._test_restore_mode(file, deg_type="multiplicative")
