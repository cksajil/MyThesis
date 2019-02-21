# Import Libraries
import numpy as np
from scipy import signal


def ANCInActionBeamer(
        Pw,
        Sw,
        x,
        xrev,
        T=10000,
        L=256,
        mu=0.0001):

    Shw = Sw
    Shx = Sw
    e_iden = np.zeros((1, T))
    Yd = xrev

    # Initiate the system
    Cx = np.zeros((1, L))   # The state of C(z)
    Cw = np.zeros((1, L))   # The weight of C(z)
    # The dummy state for the secondary path
    Sx = np.zeros((1, len(Sw)))
    # Data buffer for the control error
    e_cont = np.zeros((1, T))
    Xhx = np.zeros((1, L))   # The state of the filtered x(k)

    # And apply the FxLMS algorithm
    for k in range(0, T):
        Cx = np.roll(Cx, 1)
        Cx[0, 0] = x[k]
        Cy = np.dot(Cx, Cw[0, :])
        Sx = np.roll(Sx, 1)
        Sx[0, 0] = Cy
        e_cont[0, k] = Yd[k] - np.dot(Sx, Sw)
        Shx = np.roll(Shx, 1)
        Shx[0] = x[k]
        Xhx = np.roll(Xhx, 1)
        Xhx[0, 0] = np.dot(Shx, Shw)
        Cw = Cw + mu * e_cont[0, k] * Xhx

    return e_cont[0], Yd


def ANCInAction(Pw, Sw, x, T=10000, L=256, mu=0.0001):

    Shw = Sw
    Shx = Sw
    e_iden = np.zeros((1, T))
    Yd = signal.lfilter(Pw, 1, x) + \
        np.sqrt(0.01) * np.random.randn(1, T)

    # Initiate the system
    Cx = np.zeros((1, L))   # The state of C(z)
    Cw = np.zeros((1, L))   # The weight of C(z)
    # The dummy state for the secondary path
    Sx = np.zeros((1, len(Sw)))
    # Data buffer for the control error
    e_cont = np.zeros((1, T))
    Xhx = np.zeros((1, L))   # The state of the filtered x(k)

    # And apply the FxLMS algorithm
    for k in range(0, T):
        Cx = np.roll(Cx, 1)
        Cx[0, 0] = x[k]
        Cy = np.dot(Cx, Cw[0, :])
        Sx = np.roll(Sx, 1)
        Sx[0, 0] = Cy
        e_cont[0, k] = Yd[0, k] - np.dot(Sx, Sw)
        Shx = np.roll(Shx, 1)
        Shx[0] = x[k]
        Xhx = np.roll(Xhx, 1)
        Xhx[0, 0] = np.dot(Shx, Shw)
        Cw = Cw + mu * e_cont[0, k] * Xhx

    return e_cont[0], Yd[0]
