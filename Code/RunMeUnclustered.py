#
from SpectralFlatness import SpectralFlatness
from PerformanceMetrics import metrics
from multiprocessing import Pool
from SignalGenerator import GenerateSignal
from MyFxLMS import ANCInAction
from RIRGenKernel import ComputeRIRs
from pandas import ExcelWriter
import pandas as pd
import numpy as np
import time
print "Loading Libraries"


#

print "Setting Simulation Parameters"

c = 343        # Sound velocity (m/s)
Fs = 12000.0       # Sample frequency (samples/s)
RoomSize = [2.9, 2.14, 3.1]      # Room dimensions [x y] (m)
# Anti-source Separation distance
TransSeparation = 0.05
beta = 0.3942                             # Reverberation Time
nsample = 5000             # Number of samples
mtype = 'omnidirectional'    # Type of microphone
order = -1        # -1 equals maximum reflection order!
dim = 3        # Room dimension
orientation = 0        # Microphone orientation (rad)
hp_filter = 1        # Enable high-pass filter
T = 24000                          # Normalised Simulation Time
L = 800           # Set Filter Length
mu = 0.000001                          # Set Learning Rate
# Set MonteCarlo Iterations
MonteCarloLen = 100

#

print "Reading Constellations"

Constellation = pd.read_excel(
    open(
        'Input/Inputs__Zpos_0_82.xlsx',
        'rb'),
    index=False)
Constellation = Constellation.values

#

print "Writing Experiment Log"

with open("log.txt", "w") as logger:
    logger.write(
        "The experiment configuration were\n"
        "\nSampling Rate      					: 		%s" %
        Fs +
        "\nSound Speed        					: 		%s" %
        c +
        "\nRoom size          					: 		%s" %
        RoomSize +
        "\nRIR Samples length 					: 		%s" %
        nsample +
        "\nMicrophone Type        				: 		%s" %
        mtype +
        "\nBeta               					: 		%s" %
        beta +
        "\nNumber of Reflections  				: 		%s" %
        order +
        "\nSimulation Dimentions  				: 		%s" %
        dim +
        "\nMicrophone Orientation  			: 		%s" %
        orientation +
        "\nHigh Pass Filter        			: 		%s" %
        hp_filter +
        "\nNormalised Simulation Time        	: 		%s" %
        T +
        "\nFilter length        				: 		%s" %
        L +
        "\nLearning Rate        				: 		%s" %
        mu +
        "\nAnti Source Separation              :       %s" %
        TransSeparation +
        "\nMonte Carlo length                  :       %s" %
        MonteCarloLen)

#

print "Generating Input Signal"

x = GenerateSignal(T, Fs)

#

t1 = time.time()

print "Started Parallel Processing"


def ParallelProcessing(location):
    SourceLoc = location[0:3]
    ReceiverLoc = location[3:6]
    Antilocation = location[6:9]

    P, S = ComputeRIRs(
        c, Fs, ReceiverLoc, SourceLoc, Antilocation,
        RoomSize, beta, nsample, mtype, order, dim, orientation, hp_filter)
    PSpecFlat = SpectralFlatness(P, Fs)
    SSpecFlat = SpectralFlatness(S, Fs)
    e_cont, Yd = ANCInAction(P, S, x, T, L, mu)
    # Pd_dB, fd, Pe_dB, fe, EstimatedAttenuation, components =
    # metrics(e_cont, Yd, Fs, T)
    ANCRed = metrics(e_cont, Yd, Fs, T)
    features = np.append(
        location, [
            PSpecFlat, SSpecFlat, ANCRed])
    return features


P = Pool()
result = P.map(ParallelProcessing, Constellation)

#

print "Saving Final Result"


# cols                    =       ['Sx', 'Sy', 'Sz', 'Rx', 'Ry',
# 'Rz', 'Ax', 'Ay', 'Az', 'Attenuation', '30Hz', '60Hz', '90Hz']
cols = [
    'Sx',
    'Sy',
    'Sz',
    'Rx',
    'Ry',
    'Rz',
    'Ax',
    'Ay',
    'Az',
    'PSpectroFlat',
    'SSpectroFlat',
    'ANCRed']
data = pd.DataFrame(result, columns=cols)
writer = ExcelWriter(
    'Results/Results__Zpos_0.82_UnClustered.xlsx')


data.to_excel(writer, 'Sheet1')
writer.save()

#

t2 = time.time()

m, s = divmod(t2 - t1, 60)
h, m = divmod(m, 60)
d, h = divmod(h, 24)

print "Done Simulation in %d Days, %d Hours, %02d Minutes, %02d Seconds" % (
    d, h, m, s)

#
