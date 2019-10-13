from tkinter import Tk
from tkinter import filedialog
import tkinter, os
from ImageFileHandler import ImageFileHandler
from ImageRestorer import ImageRestorer

class HomeScreen:

    def __init__(self):
        self.image_manipulator = ImageFileHandler()
        self.image_restorer = ImageRestorer()
        self.set_up_screen()


    def set_up_screen(self):
        self.window = Tk()
        self.window.geometry("640x480")

        self.original_image_path= self.restored_image_path = "./test_images/cameraman.jpeg"
        self.original_image = self.image_manipulator.get_tkinter_image(self.original_image_path, (256, 256))
        self.restored_image = self.image_manipulator.get_tkinter_image(self.restored_image_path, (256, 256))

        self.original_image_text = tkinter.Label(self.window, text="ORIGINAL IMAGE")
        self.restored_image_text = tkinter.Label(self.window, text="RESTORED IMAGE")
        self.original_image_frame = tkinter.Label(self.window, image=self.original_image)
        self.restored_image_frame = tkinter.Label(self.window, image=self.restored_image)

        self.select_image_button = tkinter.Button(self.window, text="Browse...", command=self.browse_for_image)
        self.degradation_dropdown = tkinter.Listbox(self.window)
        self.restore_button = tkinter.Button(self.window, text="Restore!", command=self.restore)


        self.original_image_text.grid(row=0, column=0)
        self.restored_image_text.grid(row=0, column=1)
        self.original_image_frame.grid(row=1, column=0)
        self.restored_image_frame.grid(row=1, column=1)
        self.select_image_button.grid(row=2, column=0)
        self.degradation_dropdown.grid(row=2, column=1)
        self.restore_button.grid(row=2, column=2)

    def browse_for_image(self):
        currdir = os.getcwd()
        image_path = filedialog.askopenfilename(parent=self.window, initialdir=currdir, title='Please select an image')
        self.update_original_image_path(image_path)

    def restore(self):
        output_path = self.image_restorer.restore(self.original_image_path)
        self.update_restored_image_path(output_path)

    def update_original_image_path(self, path):
        if len(path) > 0:
            self.original_image = self.image_manipulator.get_tkinter_image(path, (256, 256))
            self.original_image_frame.config(image=self.original_image)
            self.original_image_path = path

    def update_restored_image_path(self, path):
        if len(path) > 0:
            self.restored_image = self.image_manipulator.get_tkinter_image(path, (256, 256))
            self.restored_image_frame.config(image=self.original_image)
            self.restored_image_path = path

    def run(self):
        self.window.mainloop()
