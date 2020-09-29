import numpy as np
import scipy as sp
import librosa

def cutter(x, sr=None):
    '''
    cutter(x) takes wave amplitude array (x) and trims the edges, to avoid vocal onsets/stops at the beginning/end.
    '''
    if sr == None:
        n1 = int(0.25 * len(x))
        n2 = int(0.75 * len(x))

        return x[n1:n2]

    else:
        mid_idx = int(len(x) // 2)  # middle index
        hlf_wdw = int(0.5 * (sr) / 2.0)  # half of 30 sec window in # of elements
        return x[mid_idx - hlf_wdw:mid_idx + hlf_wdw]


def formant_signal(x, order):
    '''
    formant_signal(x,order) takes wave amplitude array (x) and returns the LPC approximation
    (s_formant) based on a linear autoregressive model to a given specified order (order).

    s_formant can be understood as the (sourceless) amplitude due to the vocal formants of the signal x.
    '''

    a = librosa.lpc(x, order)  # (x,16)
    b = np.hstack([[0], -1 * a[1:]])
    s_formant = sp.signal.lfilter(b, [1], x)

    return s_formant


def residual(x, order, s=None):
    '''
    residual(x,order,s) subtracts the formants from the input signal x to yield the source wave (vocal cord buzz).
    '''
    if s is None:
        s_formant = formant_signal(x, order)
        return (x - s_formant)
    else:
        return (x - s)


def audio_processor(x, sr=44100, order=16):
    '''
    Take input file and return ready for CNN
    Expecting x from x, sr = librosa.load(filepath,sr=44100) #
    '''

    R_ = librosa.stft(residual(cutter(x, sr), order), n_fft=2048)  # complex matrix
    Rabs_ = np.abs(R_)
    RdB_ = librosa.amplitude_to_db(Rabs_, ref=np.max)
    RdB_expanded = np.expand_dims(RdB_, axis=2) # for keras
    RdB_ready = np.array([RdB_expanded, ]) # ready for keras

    return RdB_ready