import dippykit as dip
import numpy as np
from ImageFileHandler import ImageFileHandler
from ImageDegrader import ImageDegrader
import os
from ClusteringHandler import ClusteringHandler
from sklearn import preprocessing
import cv2
import scipy.ndimage as ndimage
import scipy.signal as signal
import matplotlib.pyplot as plt

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
        max_h_value = 40

        blocks = self.break_image_into_blocks(degraded_image, block_size)
        variances = self.get_variances_of_blocks(blocks)
        means = self.get_means_of_blocks(blocks)

        data = []
        for i in range(len(blocks)):
            data.append([means[i], variances[i]])

        output_image = np.zeros([degraded_image.shape[0], degraded_image.shape[1]])
        cluster_image = np.zeros([degraded_image.shape[0], degraded_image.shape[1]])
        ch = ClusteringHandler(data)
        ch.cluster_data()
        clustered_labels = ch.labels
        print(clustered_labels)
        cluster_centers = np.asarray(ch.cluster_centers)

        mean_centers = cluster_centers[:, 0]
        var_centers = cluster_centers[:, 1]

        h_params = np.linspace(max_h_value, max_h_value, ch.num_clusters)
        window_sizes = [21 for i in range(ch.num_clusters)]
        count = 0

        h_params = [a for _, a in sorted(zip(var_centers, h_params), reverse=True)]
        h_params = h_params * mean_centers ** (1/2) * 7

        print("H Parameters that will be used: {}".format(h_params))

        for m in range(0, degraded_image.shape[0], block_size):
            for n in range(0, degraded_image.shape[1], block_size):

                cur_percentile = clustered_labels[m//block_size*(degraded_image.shape[0]//block_size)+(n//block_size)]

                cur_block = self.create_surrounding_block((m, n), degraded_image, block_size)

                cluster_image[m:m + block_size, n:n + block_size] = cur_percentile / ch.num_clusters

                output_image[m:m + block_size, n:n + block_size] = self.fast_multiplicative_restore(
                    cur_block,
                    h_param=h_params[cur_percentile],
                    search_window_size=window_sizes[cur_percentile]
                )[block_size:block_size * 2, block_size:block_size * 2]
                count += 1
        blurred_borders_image = self.blur_borders(output_image, cluster_image)
        return blurred_borders_image, cluster_image


    def blur_borders(self, image, cluster_image):
        kernel_size = 7
        pad_size = kernel_size // 2 + 1
        kernel_maker_array = np.zeros((kernel_size,kernel_size))
        kernel_maker_array[1, 1] = 1

        kernel = ndimage.filters.gaussian_filter(kernel_maker_array, 1)
        padded_image = np.pad(image, (pad_size, pad_size), mode='symmetric')
        padded_cluster_image = np.pad(cluster_image, (pad_size, pad_size), mode='symmetric')
        blurred_image = np.zeros_like(image)
        for m in range(pad_size, image.shape[0]+pad_size):
            for n in range(pad_size, image.shape[1]+pad_size):
                surrounding_box = padded_cluster_image[m-pad_size+1:m+pad_size, n-pad_size+1:n+pad_size]
                if np.all(surrounding_box==surrounding_box[0, 0]):
                    blurred_image[m-pad_size, n-pad_size] = padded_image[m, n]
                else:
                    image_box = padded_image[m-pad_size+1:m+pad_size, n-pad_size+1:n+pad_size]
                    convolved_image = signal.convolve2d(kernel, image_box)
                    blurred_image[m-pad_size, n-pad_size] = convolved_image[pad_size, pad_size]
        return blurred_image


    def create_surrounding_block(self, origin_index, image, block_size):
        output = np.zeros((block_size*3, block_size*3))

        extended_image = np.concatenate((image, image, image), axis=1)
        extended_image = np.concatenate((extended_image, extended_image, extended_image), axis=0)

        new_origin = (origin_index[0] + image.shape[0], origin_index[1] + image.shape[1])

        output = extended_image[
                 new_origin[0] - block_size:new_origin[0] + 2*block_size,
                 new_origin[1] - block_size:new_origin[1] + 2*block_size
                 ]

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

    def _test_restore_mode(self, file, deg_type, save_images=False, name=None):
        fh = ImageFileHandler()
        im_degrader = ImageDegrader()
        im = fh.open_image_file_as_matrix(file)
        degraded_im = im_degrader.degrade(im, degradation_type=deg_type)
        restored_im, clustered_im = self.multiplicative_clustering_restore(degraded_im)

        if save_images:
            dip.im_write(dip.float_to_im(degraded_im), "./"+name+"_degraded_image.jpg", quality=95)
            dip.im_write(dip.float_to_im(restored_im), "./"+name+"_restored_image.jpg", quality=95)
            dip.im_write(dip.float_to_im(clustered_im), "./"+name+"_clustered_image.jpg", quality=96)

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

    def _plot_psnr_against_var(self, image, deg_type):
        fh = ImageFileHandler()
        im_degrader = ImageDegrader()
        im = fh.open_image_file_as_matrix(file)
        vars = np.linspace(0.01, 0.3, 20)
        restored_psnrs = []
        degraded_psnrs = []
        count = 0
        for var in vars:
            degraded_im = im_degrader.degrade(im, degradation_type=deg_type, severity_value=var)
            restored_im, _ = self.multiplicative_clustering_restore(degraded_im)
            degraded_psnrs.append(dip.PSNR(im, degraded_im))
            restored_psnrs.append(dip.PSNR(im, restored_im))
            count += 1
            print("{} out of {} complete".format(count, len(vars)))
        plt.plot(vars, restored_psnrs, "b")
        plt.plot(vars, degraded_psnrs, "r")
        plt.plot(vars, np.subtract(restored_psnrs, degraded_psnrs), "k")

        plt.show()

if __name__ == "__main__":
    file = os.path.join(os.getcwd(), "test_images", "dione.jpg")
    ir = ImageRestorer()
    #ir._test_restore_mode(file, deg_type="multiplicative", save_images=True, name="dione")
    ir._plot_psnr_against_var(file, deg_type="multiplicative")
