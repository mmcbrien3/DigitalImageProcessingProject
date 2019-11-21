import ImageFileHandler
import ImageRestorer
import ImageDegrader
import sys
from base64 import decodebytes
import base64


def get_noise(noise_string):
    noise_string = noise_string.lower()
    sev_value = 0
    if noise_string == "low":
        sev_value = 0.02
    elif noise_string == "medium":
        sev_value = 0.08
    else:
        sev_value = 0.3
    return sev_value


def convert_image(base64_image):
    base64_image = base64_image.encode()
    im_path = "image.jpg"
    with open(im_path, "wb") as f:
        f.write(decodebytes(base64_image))
    fh = ImageFileHandler.ImageFileHandler()
    im = fh.open_image_file_as_matrix(im_path)
    return fh.resize_image(im, (512, 512))


if __name__ == "__main__":
    print("HELLO from python!")
    sys.exit()
    sev_value = get_noise(sys.argv[1])
    im = convert_image(sys.argv[2])

    fh = ImageFileHandler.ImageFileHandler()
    im_degrader = ImageDegrader.ImageDegrader()
    im_restorer = ImageRestorer.ImageRestorer()

    degraded_im = im_degrader.degrade(im, degradation_type="multiplicative", severity_value=sev_value)
    restored_im, clustered_im, h_params = im_restorer.multiplicative_clustering_restore(degraded_im)
    restored_im_path = "./restored_image.jpg"
    degraded_im_path = "./degraded_image.jpg"

    fh.save_matrix_as_image_file(restored_im, restored_im_path)
    fh.save_matrix_as_image_file(degraded_im, degraded_im_path)

    with open(degraded_im_path, "rb") as f:
        print(base64.b64encode(f.read()))
    with open(restored_im_path, "rb") as f:
        print(base64.b64encode(f.read()))
