import dippykit as dip
import numpy as np
from ImageFileHandler import ImageFileHandler
import os


class ImageDegrader():

    def __init__(self):
        pass

    def degrade(self, image, degradation_type="gaussian", severity_value=0.05):

        if degradation_type.lower() == "gaussian":
            return self.add_gaussian_noise(image)
        elif degradation_type.lower() == "additive":
            return self.add_additive_noise(image)
        elif degradation_type.lower() == "multiplicative":
            return self.add_multiplicative_noise(image, var=severity_value)
        else:
            raise Exception("{} is not a valid degradation type.".format(degradation_type))

    def add_gaussian_noise(self, image):
        pass

    def add_additive_noise(self, image):
        pass

    def add_multiplicative_noise(self, image, mean=0.0, var=0.05, clip=True):

        low_clip = 0
        if image.min() < 0:
            low_clip = -1

        noise = np.random.normal(mean, var ** 0.5, image.shape)
        out = image + image * noise

        if clip:
            out = np.clip(out, low_clip, 1.0)

        return out

    def _test_noise_mode(self, file, deg_type):
        fh = ImageFileHandler()
        im = fh.open_image_file_as_matrix(file)
        degraded_im = self.degrade(im, degradation_type=deg_type)
        dip.figure()
        dip.subplot(121)
        dip.imshow(im, cmap="gray")
        dip.subplot(122)
        dip.imshow(degraded_im, cmap="gray")
        dip.xlabel("PSNR: {0:.2f}".format(dip.PSNR(im, degraded_im)))
        dip.show()


if __name__ == "__main__":
    file = os.path.join(os.getcwd(), "test_images", "cameraman.jpeg")
    id = ImageDegrader()
    id._test_noise_mode(file, deg_type="multiplicative")

