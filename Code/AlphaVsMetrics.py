#

from SignalGenerator import *
from PreProcess import normalize
import pyroomacoustics as pra
import matplotlib.pyplot as plt
from MyFxLMSB import *
from PerformanceMetrics import metrics
import pandas as pd

#


T = 10
mu = 0.0001
L = 800
rirlen = 1000

#

xtone = GenerateTonalSignal(T * 8000, 8000.0)
xtone = normalize(xtone)
Fs = 8000
silence = np.zeros_like(xtone)
result = pd.DataFrame(columns=['Alpha', 'ANCRed', 'ANCRedBeam'])


#

# Create Room, Sensors and calculate beam patterns
Lg_t = 0.100                # filter size in seconds
Lg = np.ceil(Lg_t * Fs)     # filter size in samples
alphas = np.arange(0.1, 1, 0.05)
source = np.array([1, 4.5])
interferer = np.array([3.5, 3.])
radius = 0.15

roomDim = [8, 6]
center = [1, 3.5]
fft_len = 512
echo = pra.circular_2D_array(
    center=center, M=6, phi0=0, radius=radius)
echo = np.concatenate(
    (echo, np.array(center, ndmin=2).T), axis=1)


for alpha in alphas:
    room_bf = pra.ShoeBox(
        roomDim,
        fs=Fs,
        max_order=64,
        absorption=alpha)
    mics = pra.Beamformer(echo, room_bf.fs, N=fft_len, Lg=Lg)
    room_bf.add_microphone_array(mics)
    room_bf.add_source(source, delay=0., signal=xtone)
    room_bf.add_source(interferer, delay=0, signal=silence)

    # Compute DAS weights
    mics.rake_delay_and_sum_weights(room_bf.sources[0][:1])

#
# Do Beamforming
    room_bf.image_source_model(use_libroom=True)
    room_bf.compute_rir()
    room_bf.simulate()

#
    signal_das = mics.process(FD=False)
    signal_das = normalize(signal_das)
    centerMicSignal = room_bf.mic_array.signals[-1, :]
    centerMicSignal = normalize(centerMicSignal)

#
    P, S = room_bf.rir[0]
    P = P[:rirlen]
    P = normalize(P)
    S = S[:rirlen]
    S = normalize(S)
    error_b, Yd_b = ANCInActionBeamer(
        P, S, xtone, centerMicSignal, T * 8000, L, mu)
    error, Yd = ANCInAction(P, S, xtone, T * 8000, L, mu)
    ANCRedB = metrics(error_b, Yd_b, Fs, T * 8000)
    ANCRed = metrics(error, Yd, Fs, T * 8000)

    result = result.append({'Alpha': alpha,
                            'ANCRed': ANCRed,
                            'ANCRedBeam': ANCRedB},
                           ignore_index=True)
    print alpha, ANCRed, ANCRedB

# plot the room and resulting beamformer before simulation
fig, ax = room_bf.plot(freq=[300, 450, 500], img_order=1)
ax.legend(['300', '450', '500'])
plt.xlabel('x')
plt.ylabel('y')
plt.show()

fig.savefig('Images/DSRoom' +
            str(roomDim[0]) +
            'by' +
            str(roomDim[1]) +
            '.png', dpi=300)
result.to_csv('Results/MasterData' +
              str(roomDim[0]) +
              'by' +
              str(roomDim[1]) +
              '.csv', sep=',', index=False)
