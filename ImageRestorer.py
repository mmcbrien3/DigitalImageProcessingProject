import dippykit as dip
import numpy as np
from sklearn import preprocessing
from scipy import ndimage
import scipy.stats as sci_stats
from ImageFileHandler import ImageFileHandler
from ImageDegrader import ImageDegrader
import os
from ClusteringHandler import ClusteringHandler
from sklearn import preprocessing
import cv2
import scipy.ndimage as ndimage
import scipy.signal as signal
import matplotlib.pyplot as plt
from multiprocessing import Process, Manager

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

        max_h_value = 30

        blocks = self.break_image_into_blocks(degraded_image, block_size)
        variances = self.get_variances_of_blocks(blocks)
        third_moments = self.get_statistic_of_blocks(blocks, lambda b: sci_stats.moment(b.flatten(), moment=3))
        medians = self.get_statistic_of_blocks(blocks, np.median)
        means = self.get_means_of_blocks(blocks)
        maxs = self.get_statistic_of_blocks(blocks, lambda b: np.percentile(b, 55))
        mins = self.get_statistic_of_blocks(blocks, lambda b: np.percentile(b, 45))

        scaler = preprocessing.StandardScaler()
        data = []
        for i in range(len(blocks)):
            data.append([variances[i], means[i], medians[i], maxs[i], mins[i]])

        scaler.fit(data)

        output_image = np.zeros([degraded_image.shape[0], degraded_image.shape[1]])
        cluster_image = np.zeros([degraded_image.shape[0], degraded_image.shape[1]])
        h_param_image = np.zeros(degraded_image.shape)
        ch = ClusteringHandler(data)
        ch.cluster_data()
        clustered_labels = ch.labels
        print(clustered_labels)
        cluster_centers = np.asarray(ch.cluster_centers)
        print(cluster_centers)

        print(cluster_centers)
        max_c = cluster_centers[:, 3]
        min_c = cluster_centers[:, 4]
        mean_c = cluster_centers[:, 1]
        var_centers = cluster_centers[:, 0]

        h_params = np.linspace(max_h_value, max_h_value, ch.num_clusters)
        window_sizes = [21 for i in range(ch.num_clusters)]
        count = 0

        h_params = [a for _, a in sorted(zip(var_centers, h_params), reverse=True)]
        h_params = var_centers * 15000 * mean_c * (max_c - min_c)

        print("VAR CENTERS: {}".format(var_centers))
        print("MEAN CENTERS: {}".format(0))
        print("H Parameters that will be used: {}".format(h_params))

        for m in range(0, degraded_image.shape[0], block_size):
            for n in range(0, degraded_image.shape[1], block_size):
                cur_percentile = clustered_labels[
                    m // block_size * (degraded_image.shape[0] // block_size) + (n // block_size)]
                cluster_image[m:m + block_size, n:n + block_size] = cur_percentile / ch.num_clusters
                h_param_image[m:m + block_size, n:n + block_size] = h_params[cur_percentile]

        h_param_image = self.blur_borders(h_param_image, cluster_image)
        print("number of h params to be used: {}".format(np.size(np.unique(h_param_image.flatten()))))

        h_param_count = 0
        all_temp_outputs = []
        h_param_list = np.unique(h_param_image.flatten())

        processes = []
        manager = Manager()
        return_dict = manager.dict()
        blocked_pad_size = 32
        blocked_image = self.create_surrounding_block_fast(degraded_image, pad_width=blocked_pad_size)

        process_count = 0
        for c in h_param_list:
            temp_output_image = self.fast_multiplicative_restore(
                blocked_image,
                h_param=c,
                search_window_size=21
            )[blocked_pad_size:degraded_image.shape[0]+blocked_pad_size, blocked_pad_size:degraded_image.shape[1]+blocked_pad_size]
            return_dict[c] = temp_output_image
            h_param_count += 1
            print("{} h param finished.".format(h_param_count))
        print('finished restores')


        print(return_dict.keys())
        output_image = output_image.flatten()
        h_param_image = h_param_image.flatten()
        for cur_param in return_dict.keys():
            idx = h_param_image == cur_param
            return_dict[cur_param] = return_dict[cur_param].flatten()
            output_image[idx] = return_dict[cur_param][idx]
        return dip.im_to_float(dip.float_to_im(output_image.reshape(degraded_image.shape))), cluster_image, h_params


    def perform_single_h_param(self, cur_block, h, index, output_array):
        temp_output_image = self.fast_multiplicative_restore(
            cur_block,
            h_param=h,
            search_window_size=21
        )
        output_array[h] = temp_output_image

        print("Finished single h param")

    def blur_borders(self, image, cluster_image):
        min_h_param = np.min(image)
        max_h_param = np.max(image)
        num_h_params = np.size(np.unique(cluster_image.flatten()))*3
        kernel_size = 41

        blurred_image = cv2.GaussianBlur(image, ksize=(kernel_size, kernel_size), sigmaX=0)
        quantized_image = np.zeros_like(image)
        quantize_values = np.linspace(min_h_param, max_h_param, num_h_params)
        known_quantized = {}
        for m in range(blurred_image.shape[0]):
            for n in range(blurred_image.shape[1]):

                best_dist = np.inf
                best_val = -1
                cur_dist = -1
                prev_dist = np.inf
                cur_val = blurred_image[m, n]
                if cur_val in known_quantized.keys():
                    quantized_image[m, n] = known_quantized[cur_val]
                    continue
                for q in quantize_values:
                    cur_dist = np.abs(cur_val - q)

                    if cur_dist < best_dist:
                        best_dist = cur_dist
                        best_val = q
                    if cur_dist > prev_dist:
                        break
                    prev_dist = cur_dist

                if cur_val not in known_quantized.keys():
                    known_quantized[cur_val] = best_val
                quantized_image[m, n] = best_val
        return quantized_image


    def create_surrounding_block_fast(self, image, pad_width=32):

        extended_image = np.concatenate((image, image, image), axis=1)
        extended_image = np.concatenate((extended_image, extended_image, extended_image), axis=0)

        return extended_image[image.shape[0]-pad_width:image.shape[0]*2+pad_width, image.shape[1]-pad_width:image.shape[1]*2+pad_width]

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

    def get_maxes_of_blocks(self, blocks):
        return self.get_statistic_of_blocks(blocks, np.max)

    def get_mins_of_blocks(self, blocks):
        return self.get_statistic_of_blocks(blocks, np.min)

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
        degraded_im = im_degrader.degrade(im, degradation_type=deg_type, severity_value=.5)
        restored_im, clustered_im, _ = self.multiplicative_clustering_restore(degraded_im)

        if save_images:
            dip.im_write(dip.float_to_im(degraded_im), "./"+name+"_degraded_image.jpg", quality=95)
            dip.im_write(dip.float_to_im(restored_im), "./"+name+"_restored_image.jpg", quality=95)
            dip.im_write(dip.float_to_im(clustered_im), "./"+name+"_clustered_image.jpg", quality=95)

        dip.figure()
        dip.subplot(131)
        dip.imshow(im, cmap="gray")
        dip.subplot(132)
        dip.imshow(degraded_im, cmap="gray")
        dip.xlabel("PSNR: {0:.2f}, SSIM: {1:.2f}".format(dip.PSNR(im, degraded_im), dip.SSIM(im, degraded_im)[0]))
        dip.subplot(133)
        dip.imshow(restored_im, cmap="gray")
        dip.xlabel("PSNR: {0:.2f}, SSIM: {1:.2f}".format(dip.PSNR(im, restored_im), dip.SSIM(im, restored_im)[0]))
        dip.show()

    def _plot_psnr_against_var(self, image, deg_type):
        fh = ImageFileHandler()
        im_degrader = ImageDegrader()
        im = fh.open_image_file_as_matrix(file)
        vars = np.linspace(0.01, 0.5, 10)
        restored_psnrs = []
        degraded_psnrs = []
        generic_psnrs = []
        count = 0
        comparison_func = lambda x, y: dip.SSIM(x, y)[0]
        for var in vars:
            degraded_im = im_degrader.degrade(im, degradation_type=deg_type, severity_value=var)
            restored_im, _, h_params = self.multiplicative_clustering_restore(degraded_im)
            restored_generic_im = self.fast_multiplicative_restore(degraded_im, h_param=int(np.mean(h_params)), search_window_size=21)
            degraded_psnrs.append(comparison_func(im, degraded_im))
            restored_psnrs.append(comparison_func(im, restored_im))
            generic_psnrs.append(comparison_func(im, restored_generic_im))
            count += 1
            print("{} out of {} complete".format(count, len(vars)))
        plt.plot(vars, restored_psnrs, "b", label="Clustering Restore")
        plt.plot(vars, degraded_psnrs, "r", label="Degraded Image")
        plt.plot(vars, generic_psnrs, "y", label="fastN1Means Restore")
        plt.legend()
        plt.xlabel("Variance level of noise")
        plt.ylabel("SSIM")
        plt.title("Effect of clustering on SSIM")
        plt.show()

if __name__ == "__main__":

    file = os.path.join(os.getcwd(), "test_images", "tennis.jpg")
    ir = ImageRestorer()
    #ir._test_restore_mode(file, deg_type="multiplicative", save_images=True, name="lena")
    ir._plot_psnr_against_var(file, deg_type="multiplicative")
