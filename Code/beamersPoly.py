import numpy as np
from PreProcess import normalize
from SignalGenerator import *
import matplotlib.pyplot as plt
import pyroomacoustics as pra
import pandas as pd
from PerformanceMetrics import metrics
from TimeDomainDynamics import TimeDynamics
from MyFxLMS import *
# from scipy import signal
# from scipy.io import wavfile


# Simulation parameters

T = 10
mu = 0.001
L = 800
rirlen = 1000

xtone = GenerateTonalSignal(T * 8000, 8000.0)
xtone = normalize(xtone)

Fs = 8000
silence = np.zeros_like(xtone)
result = pd.DataFrame(columns=['Alpha', 'ANCRed', 'ANCRedBeam'])

Lg_t = 0.100                # Filter size in seconds
Lg = np.ceil(Lg_t * Fs)       # Filter size in samples
# alphas              =     np.arange(0.1,1,0.05)
alphas = [0.7, 0.75]
source = np.array([1, 1])
interferer = np.array([3, 2])
radius = 0.15


center = [1, 2]
fft_len = 512
echo = pra.circular_2D_array(
    center=center, M=6, phi0=0, radius=radius)
echo = np.concatenate(
    (echo, np.array(center, ndmin=2).T), axis=1)
sigma2_n = 5e-7
max_order_design = 1


for alpha in alphas:
    corners = np.array(
        [[0, 0], [0, 4], [6, 4], [6, 1], [2, 1], [2, 0]]).T  # [x,y]
    roomPoly = pra.Room.from_corners(
        corners, fs=Fs, max_order=12, absorption=alpha)
    mics = pra.Beamformer(echo, Fs, N=fft_len, Lg=Lg)
    roomPoly.add_microphone_array(mics)
    roomPoly.add_source(source, delay=0, signal=xtone)
    roomPoly.add_source(interferer, delay=0, signal=silence)
    roomPoly.image_source_model(use_libroom=True)
    roomPoly.compute_rir()
    roomPoly.simulate()

    # Rake MVDR simulation
    BeamformerType = 'RakeMVDR'
    good_sources = roomPoly.sources[0][:max_order_design + 1]
    bad_sources = roomPoly.sources[1][:max_order_design + 1]
    mics.rake_mvdr_filters(good_sources,
                           bad_sources,
                           sigma2_n * np.eye(mics.Lg * mics.M))
    output = mics.process()
    out = pra.normalize(pra.highpass(output, Fs))
    out = normalize(out)

    # Rake Perceptual simulation
    # BeamformerType = 'RakePerceptual'
    # good_sources = room1.sources[0][:max_order_design+1]
    # bad_sources = room1.sources[1][:max_order_design+1]
    # mics.rake_perceptual_filters(good_sources,
    #                     bad_sources,
    #                     sigma2_n*np.eye(mics.Lg*mics.M))
    # output          =   mics.process()
    # out             =   pra.normalize(pra.highpass(output, Fs))

    # input_mic       =   pra.normalize(pra.highpass(mics.signals[mics.M//2], Fs))
    # input_mic       =   normalize(input_mic)

    P, S = roomPoly.rir[0]
    P = P[:rirlen]
    P = normalize(P)
    S = S[:rirlen]
    P = normalize(S)

    error_b, Yd_b = ANCInActionBeamer(
        P, S, xtone, out, T * 8000, L, mu)
    error, Yd = ANCInAction(P, S, xtone, T * 8000, L, mu)
    Pd_dBB, fdB, Pe_dBB, feB, ANCRedB = metrics(
        error_b, Yd_b, Fs, T * 8000)
    Pd_dB, fd, Pe_dB, fe, ANCRed = metrics(
        error, Yd, Fs, T * 8000)

    result = result.append({'Alpha': alpha,
                            'ANCRed': ANCRed,
                            'ANCRedBeam': ANCRedB},
                           ignore_index=True)
    print alpha, ANCRed, ANCRedB

TimeDynamics(Pd_dBB, fdB, Pe_dBB, feB, ANCRedB)


fig, ax = roomPoly.plot(freq=[280, 320, 350], img_order=1)
ax.legend(['300', '450', '500'])
plt.xlabel('x')
plt.ylabel('y')
plt.show()

fig.savefig('Images/' + BeamformerType + 'RoomL.png', dpi=300)
result.to_csv(
    'Results/' +
    BeamformerType +
    'RoomL.csv',
    sep=',',
    index=False)
