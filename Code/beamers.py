import numpy as np
from PreProcess import normalize
from SignalGenerator import *
import matplotlib.pyplot as plt
import pyroomacoustics as pra
import pandas as pd
from PerformanceMetrics import metrics
from TimeDomainDynamics import TimeDynamics
from MyFxLMS import *
from scipy import signal
from scipy.io import wavfile
from Designer import ANCTime

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
alphas = [0.25, 0.3]
source = np.array([3, 4])
interferer = np.array([1, 5])
radius = 0.15


roomDim = [4, 6]
center = [2.1, 4.3]
fft_len = 512
echo = pra.circular_2D_array(
    center=center, M=6, phi0=0, radius=radius)
echo = np.concatenate(
    (echo, np.array(center, ndmin=2).T), axis=1)
sigma2_n = 5e-7
max_order_design = 1


for alpha in alphas:
    room1 = pra.ShoeBox(
        roomDim,
        fs=Fs,
        max_order=64,
        absorption=alpha)
    mics = pra.Beamformer(echo, Fs, N=fft_len, Lg=Lg)
    room1.add_microphone_array(mics)
    room1.add_source(source, delay=0, signal=xtone)
    room1.add_source(interferer, delay=0, signal=silence)
    room1.image_source_model(use_libroom=True)
    room1.compute_rir()
    room1.simulate()

    # Rake MVDR simulation
    BeamformerType = 'RakeMVDR'
    good_sources = room1.sources[0][:max_order_design + 1]
    bad_sources = room1.sources[1][:max_order_design + 1]
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

    input_mic = pra.normalize(
        pra.highpass(mics.signals[mics.M // 2], Fs))
    input_mic = normalize(input_mic)

    P, S = room1.rir[0]
    P = P[:rirlen]
    P = normalize(P)
    S = S[:rirlen]
    P = normalize(S)

    # wavfile.write('Results/'+'IRPrimary'+BeamformerType+'.wav',
    # Fs,  P)

    error_b, Yd_b = ANCInActionBeamer(
        P, S, xtone, out, T * 8000, L, mu)
    error, Yd = ANCInAction(P, S, xtone, T * 8000, L, mu)
    Pd_dBB, fdB, Pe_dBB, feB, ANCRedB = metrics(
        error_b, Yd_b, Fs, T * 8000)
    Pd_dB, fd, Pe_dB, fe, ANCRed = metrics(
        error, Yd, Fs, T * 8000)

    # result = result.append({'Alpha': alpha, 'ANCRed':ANCRed,
    # 'ANCRedBeam':ANCRedB}, ignore_index=True)
    print alpha, ANCRed, ANCRedB

TimeDynamics(Pd_dBB, fdB, Pe_dBB, feB, ANCRedB)


fig, ax = room1.plot(freq=[280, 320, 350], img_order=1)
ax.legend(['300', '450', '500'])
plt.xlabel('x')
plt.ylabel('y')
plt.show()

fig.savefig('Images/Room_' +
            BeamformerType +
            str(roomDim[0]) +
            'by' +
            str(roomDim[1]) +
            '.png', dpi=300)
result.to_csv('Results/' +
              BeamformerType +
              str(roomDim[0]) +
              'by' +
              str(roomDim[1]) +
              '.csv', sep=',', index=False)
