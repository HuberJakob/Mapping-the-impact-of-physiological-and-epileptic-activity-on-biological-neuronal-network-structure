# This script is based on the "CALCULATION OF CALCIUM SIGNALS (∆F/F) FROM A REGION oF INTEREST" [1, p. 33]
# [1]   H. Jia, N. L. Rochefort, X. Chen, and A. Konnerth, “In vivo two-photon
#       imaging of sensory-evoked dendritic calcium signals in cortical neurons,”
#       Nature Protocols, vol. 6, no. 1, pp. 28–35, Jan. 2011. doi: 10.1038/nprot.
#       2010.169.
# ChatGPT (OpenAI) and Claude (Antrophic) were partly used as assistance for code generation which was reviewed and verified by the creator
import numpy as np

def cumtrapz_numpy(y):
    """Cumulative trapezoidal integral along 1D array."""
    y = np.asarray(y)
    cum = np.zeros_like(y, dtype=float)
    cum[1:] = np.cumsum((y[1:] + y[:-1]) / 2)
    return cum


def relative_fluorescence(meanstack):
    """
    meanstack: array of shape (ROI, time)
    returns: relative fluorescence dF_F array of same shape
    """

    n_n, n_frames = meanstack.shape

    #  timing parameters
    t0_s = 1 # exponential decay time constant in seconds
    t1_s = 2 # smoothing window size in seconds
    t2_s = 4 # baseline window size in seconds
    f = 4    # Framerate 
    # convert seconds into frames with framerate f
    t0 = int(round(t0_s * f))
    t1 = int(round(t1_s * f))
    t2 = int(round(t2_s * f))

    # initialize arrays 
    td_baseline = np.zeros_like(meanstack, dtype=float)
    dF_F = np.zeros_like(meanstack, dtype=float)
    F_bar = np.zeros_like(meanstack, dtype=float)
    rel_f = np.zeros_like(meanstack, dtype=float)

    # main loop
    for cell in range(n_n): # iterate trough ROIs

        # Smooth fluorescence (F_bar)
        for frame in range(t1, n_frames - t1):
            left = frame - round(t1 / 2)
            right = frame + round(t1 / 2) + 1  

            F_bar[cell, frame] = (
                1 / t1 * np.trapezoid(meanstack[cell, left:right]) # smoothing with trapezoidal integration of the moving smoothing window
            )

        # Baseline F0 - minumum of the time-dependent baseline window
        for frame in range(t2 + 2, F_bar.shape[1]):
            td_baseline[cell, frame] = np.min(
                F_bar[cell, frame - t2 - 1:frame]
            )

        # Relative fluorescence (dF/F0) compared to the baseline fluorescence
        for frame in range(n_frames):
            if td_baseline[cell, frame] != 0:
                rel_f[cell, frame] = (
                    (meanstack[cell, frame] - td_baseline[cell, frame])
                    / td_baseline[cell, frame]
                )
            else:
                rel_f[cell, frame] = 0

    #  edge handling
    rel_f[:, :t2 + t1] = 0
    rel_f[:, -t1:] = 0
    
    # parameter for exponential weighted moving average for noise filtering
    n_tau = np.arange(0, n_frames + 1)
    w = np.exp(-np.abs(n_tau) / t0)
    w_integral = cumtrapz_numpy(w) # cumulative trapezoidal integration 

    # exponential weighted moving average for noise filtering for each ROI
    for cell in range(n_n):
        
        for frame in range(n_frames):
            S = rel_f[cell, frame::-1] * w[:frame + 1]
            dF_F[cell, frame] = np.trapezoid(S) / w_integral[frame + 1]
        
        print(f"Prepared fluorescence trace for cell {cell + 1} of {n_n}")

    print("Fluorescence traces prepared")

    return dF_F
    
    print("Fluorescence traces prepared")
    return rel_f

if __name__ == "__main__":

    relative_fluorescence()
