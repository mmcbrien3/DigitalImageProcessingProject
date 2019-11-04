import dippykit as dip
import numpy as np
from ImageFileHandler import ImageFileHandler
from ImageDegrader import ImageDegrader
import os
from ClusteringHandler import ClusteringHandler
from sklearn import preprocessing
import cv2


class ImageRestorer:

    def __init__(self):
        pass

    def restore(self, image):
        return image

    def param_search_multiplicative_restore(self, degraded_image, original_image):
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

    def fast_multiplicative_restore(self, degraded_image, h_param=35, search_window_size=51):
        int_image = dip.float_to_im(degraded_image)
        return dip.im_to_float(cv2.fastNlMeansDenoising(int_image, h=h_param, searchWindowSize=search_window_size))

    def multiplicative_statistical_restore(self, degraded_image):
        block_size = 32
        blocks = self.break_image_into_blocks(degraded_image, block_size)
        variances = self.get_variances_of_blocks(blocks)

        output_image = np.zeros(degraded_image.shape)

        h_params = [35, 35, 30, 22]
        window_sizes = [41 for i in range(4)]
        count = 0
        h_params.reverse()
        for m in range(0, degraded_image.shape[0], block_size):
            for n in range(0, degraded_image.shape[1], block_size):

                cur_percentile = 0
                if variances[count] > np.percentile(variances, 25):
                    cur_percentile = 1
                if variances[count] > np.percentile(variances, 50):
                    cur_percentile = 2
                if variances[count] > np.percentile(variances, 90):
                    cur_percentile = 3

                cur_block = self.create_surrounding_block((m, n), degraded_image, block_size)

                output_image[m:m+block_size, n:n+block_size] = self.fast_multiplicative_restore(
                    cur_block,
                    h_param=h_params[cur_percentile],
                    search_window_size=window_sizes[cur_percentile]
                )[block_size:block_size*2, block_size:block_size*2]
                count += 1
        return output_image

    def multiplicative_clustering_restore(self, degraded_image):
        block_size = 16

        min_h_value = 10
        max_h_value = 30

        blocks = self.break_image_into_blocks(degraded_image, block_size)
        variances = self.get_variances_of_blocks(blocks)
        means = self.get_means_of_blocks(blocks)

        data = []
        for i in range(len(blocks)):
            data.append([means[i], variances[i]])

        output_image = np.zeros([degraded_image.shape[0], degraded_image.shape[1]])
        ch = ClusteringHandler(data)
        ch.cluster_data()
        clustered_labels = ch.labels
        print(clustered_labels)

        h_params = np.linspace(max_h_value, min_h_value, ch.num_clusters)
        window_sizes = [41 for i in range(4)]
        count = 0
        for m in range(0, degraded_image.shape[0], block_size):
            for n in range(0, degraded_image.shape[1], block_size):

                cur_percentile = clustered_labels[m//block_size*(degraded_image.shape[0]//block_size)+(n//block_size)]

                cur_block = self.create_surrounding_block((m, n), degraded_image, block_size)

                # output_image[m:m + block_size, n:n + block_size, :] = 0
                # if cur_percentile == 0:
                #     x = 3
                # output_image[m:m + block_size, n:n + block_size, cur_percentile] = 1

                output_image[m:m + block_size, n:n + block_size] = self.fast_multiplicative_restore(
                    cur_block,
                    h_param=h_params[cur_percentile],
                    search_window_size=window_sizes[cur_percentile]
                )[block_size:block_size * 2, block_size:block_size * 2]
                count += 1
        return output_image


    def create_surrounding_block(self, origin_index, image, block_size):
        output = np.zeros((block_size*3, block_size*3))

        extended_image = np.concatenate((image, image, image), axis=1)
        extended_image = np.concatenate((extended_image, extended_image, extended_image), axis=0)

        new_origin = (origin_index[0] + image.shape[0], origin_index[1] + image.shape[1])

        output = extended_image[new_origin[0] - block_size:new_origin[0] + 2*block_size, new_origin[1] - block_size:new_origin[1] + 2*block_size]

        return output


    def break_image_into_blocks(self, image, block_size):

        if image.shape[0] % block_size != 0 or image.shape[1] % block_size != 0:
            raise Exception("Block size of {} does not evenly divide into the image shape of {}".format(block_size, image.shape))

        blocks = []

        for m in range(0, image.shape[0], block_size):
            for n in range(0, image.shape[1], block_size):
                blocks.append(image[m:m+block_size, n:n+block_size])

        return blocks

    def get_statistic_of_blocks(self, blocks, statistics_function):
        stats = []
        for b in blocks:
            stats.append(statistics_function(b))
        return stats


    def get_variances_of_blocks(self, blocks):
        return self.get_statistic_of_blocks(blocks, np.var)

    def get_means_of_blocks(self, blocks):
        return self.get_statistic_of_blocks(blocks, np.mean)


    def slow_multiplicative_restore(self, degraded_image):
        h = 15
        s_window_size = 21
        t_window_size = 7

    def calc_d_squared(self, im, p, q, f):
        scaling_factor = 1 / np.square(2 * f + 1)
        p_min = np.max([p - f, 0])
        q_min = np.max([q - f, 0])
        p_max = np.min([p + f, np.shape[0]])
        im_p_vec = 0
        pass

    def calc_weights(self, d_squared, sigma_squared, h):
        max_val = np.max([d_squared - 2*sigma_squared, 0.0])
        return np.exp(-max_val / np.square(h))

    def _test_restore_mode(self, file, deg_type):
        fh = ImageFileHandler()
        im_degrader = ImageDegrader()
        im = fh.open_image_file_as_matrix(file)
        degraded_im = im_degrader.degrade(im, degradation_type=deg_type)
        restored_im = self.multiplicative_clustering_restore(degraded_im)
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
