from PIL import Image, ImageTk
import dippykit as dip


class ImageFileHandler:

    def __init__(self):
        pass

    def get_tkinter_image(self, image_path, size):
        img = Image.open(image_path)
        img = img.resize(size, Image.ANTIALIAS)
        return ImageTk.PhotoImage(img)

    def rgb_to_gray(self, image):
        if len(image.shape) != 3:
            print("Not an RGB image, returning.")
            return image
        else:
            r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
            return 0.2989 * r + 0.5870 * g + 0.1140 * b

    def open_image_file_as_matrix(self, image_path):
        im = dip.im_read(image_path)
        float_im = dip.im_to_float(im)
        gray_im = self.rgb_to_gray(float_im)
        return gray_im

    def resize_image(self, im, dims):
        return dip.resize(im, dims)

    def save_matrix_as_image_file(self, matrix, image_path):
        dip.im_write(dip.float_to_im(matrix), image_path)
