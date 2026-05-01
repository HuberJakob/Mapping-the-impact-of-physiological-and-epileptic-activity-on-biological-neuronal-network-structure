# This script is based on the Feature point tracking algorithm of Sbalzarini and Koumoutsakos [1].
# [1]   I. F. Sbalzarini and P. Koumoutsakos, “Feature point tracking and trajectory analysis for video imaging in cell biology,” Journal of Structural
#       Biology, vol. 151, no. 2, pp. 182–195, Aug. 2005. doi: 10.1016/j.jsb.
#       005.06.002.
# ChatGPT (OpenAI) and Claude (Antrophic) were partly used as assistance for code generation which was reviewed and verified by the creator.

# import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.ndimage import grey_dilation
from scipy.ndimage import uniform_filter
from skimage.exposure import histogram
import tifffile
# set global plot parameter
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "legend.title_fontsize": 12
})

def peakdetect_sbalzarini(img, w, pth, show_img = False):
    # Correlation length of camera noise
    lambdan = 1

    # Normalize image
    img = img.astype(float)
    img = (img - img.min()) / (img.max() - img.min())

    # set Kernel parameter accoring to the user defined kernel radius w
    idx = np.arange(-w, w + 1) 
    dm = 2 * w + 1 # kernel diameter
    # Kernel coordinates
    im = np.tile(idx[:, None], (1, dm))
    jm = np.tile(idx[None, :], (dm, 1))
    imjm2 = im**2 + jm**2 # squared euclidean distance from the kernel center

    siz = img.shape

    # ==============================================================
    # STEP 1: Image restoration
    # ==============================================================
    # normalization constant
    B = np.sum(np.exp(-(idx**2 / (4 * lambdan**2)))) 
    B = B**2

    K0 = (1 / B) * np.sum(np.exp(-(idx**2 / (2 * lambdan**2))))**2 - (B / (dm**2)) # normalization constant

    K = (np.exp(-(imjm2 / (4 * lambdan**2))) / B - (1 / (dm**2))) / K0 # final gaussian convolution kernel

    filtered = convolve2d(img, K, mode="same") #convolute the image with the convolution kernel

    # ==============================================================
    # STEP 2: Locating particles
    # ==============================================================

    pth = 0.01 * pth    # user defined percentile of the intensity histogram
    m, n = filtered.shape
    # extract histogram values
    hist_vals, bin_centers = histogram(filtered[w:m - w, w:n - w]) 
    l = len(hist_vals)
    # filter histogram for the defined percentile of pixel
    k = 1
    while np.sum(hist_vals[l - k:]) / np.sum(hist_vals) < pth:
        k += 1

    thresh = bin_centers[l - k]

    # Circular mask of the kernel diameter
    mask = np.zeros((dm, dm))
    mask[imjm2 <= w * w] = 1

    # Local maxima detection as brightes pixel of the pixel neighborhoud inside of the circular mask
    dil = grey_dilation(filtered, footprint=mask)
    Rp, Cp = np.where((dil - filtered) == 0)

    temp_r = Rp.copy()
    temp_c = Cp.copy()

    Rp = []
    Cp = []
    # Remove regions too close to the image border
    for r, c in zip(temp_r, temp_c):
        if (r > w and c > w and
            r < img.shape[0] - w and
            c < img.shape[1] - w):
            Rp.append(r)
            Cp.append(c)

    Rp = np.array(Rp)
    Cp = np.array(Cp)

    particles = np.zeros(siz)
    # threshold to remove noise related maxima
    valid = filtered[Rp, Cp] > thresh
    R = Rp[valid]
    C = Cp[valid]

    particles[R, C] = 1

    columnsInImage, rowsInImage = np.meshgrid(
        np.arange(img.shape[1]),
        np.arange(img.shape[0])
    )
    # extract the region properties of each ROI inside of a small circle around the detected maxima
    radius = 7
    region_properties = []

    for i in range(len(R)):
        centroid_x = C[i]
        centroid_y = R[i]

        circle = ((rowsInImage - centroid_y)**2 +
                  (columnsInImage - centroid_x)**2) <= radius**2

        pixel_idx_list = np.where(circle)

        mean_intensity = np.mean(img[pixel_idx_list])

        region_properties.append({
            "Centroid": (centroid_x, centroid_y),
            "Circle": circle,
            "PixelIdxList": pixel_idx_list,
            "Mean intensity": mean_intensity
        })

    # Moving average filter for visualization
    img_f = uniform_filter(img, size=5)
    if show_img:
    # Visualization
        plt.figure(figsize=(8, 6))
        plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 22,
        "axes.labelsize": 18,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "legend.title_fontsize": 8
        })
        plt.imshow(img_f, cmap="viridis")
        plt.scatter(C, R,alpha=0.2, c="orange", s=6)
        plt.title(f"{len(region_properties)} found ROIs")
        plt.xlabel("Pixel (x)")
        plt.ylabel("Pixel (y)")
        plt.axis("image")

        plt.show()
    # assign numbers to the ROIs
    control_img = np.zeros_like(img)
    for idx, region in enumerate(region_properties, start=1):
        control_img[region["PixelIdxList"]] = idx

    return region_properties

if __name__ == "__main__":
    peakdetect_sbalzarini()

    
