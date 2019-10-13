from PIL import Image, ImageTk
import dippykit as dip


class ImageFileHandler:

    def __init__(self):
        pass

    def get_tkinter_image(self, image_path, size):
        img = Image.open(image_path)
        img = img.resize(size, Image.ANTIALIAS)
        return ImageTk.PhotoImage(img)

    def open_image_file_as_matrix(self, image_path):
        return dip.im_to_float(dip.im_read(image_path))

    def save_matrix_as_image_file(self, matrix, image_path):
        dip.im_write(dip.float_to_im(matrix), image_path)
