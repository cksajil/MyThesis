from scipy import signal
import numpy as np

offset = 1e-20


def SpectralFlatness(x, Fs):
    f, Pxx_den = signal.periodogram(x, Fs)
    spectralflatness = np.exp(
        np.mean(np.log(Pxx_den + offset))) / np.mean(Pxx_den)
    return spectralflatness
    # plt.semilogy(f, Pxx_den)
    # plt.ylim([1e-7, 1e2])
    # plt.xlabel('frequency [Hz]')
    # plt.ylabel('PSD [V**2/Hz]')
    # plt.show()
