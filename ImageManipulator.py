from PIL import Image, ImageTk

class ImageManipulator:

    def __init__(self):
        pass

    def get_tkinter_image(self, image_path, size):
        img = Image.open(image_path)
        img = img.resize(size, Image.ANTIALIAS)
        return ImageTk.PhotoImage(img)