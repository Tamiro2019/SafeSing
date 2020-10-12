import numpy as np
import librosa
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
import os

rc('font', weight='bold')
rcParams['text.latex.preamble'] = r'\usepackage{sfmath} \boldmath'

matplotlib.use('Agg')
plt.rcParams["figure.figsize"] = (6, 3)

def make_plot_wave(wave,path,sr=44100):

    time = (1.0/sr)*np.arange(len(wave))
    plt.clf()
    plt.plot(time,wave)
    plt.title("Your Audio Wave")
    plt.xlabel('time [s]')
    plt.ylabel('amplitude [arb.]')

    #plt.savefig(os.path.join(path, 'wave.png'))
    plt.savefig(os.path.join(path, 'wave.svg'), format='svg', dpi=300)

    return 'wave.svg'

def make_plot_pitches(wave,chunk_labels,chunk_size,class_array,path,sr=44100):

    plt.clf()

    f0, voiced_flag, voiced_probs = librosa.pyin(y=wave, sr=sr, fmin=librosa.note_to_hz('C3'), fmax=librosa.note_to_hz('C5'))
    times = librosa.times_like(f0, sr=44100)

    max_f = np.nanmax(f0)
    min_f = np.nanmin(f0)

    ref_values = 261.3 * (2.0 ** np.linspace(-2.0, 2.0, 2 * 24))
    ref_ticks = librosa.hz_to_note(261.3 * (2.0 ** np.linspace(-2.0, 2.0, 2 * 24)))

    plt.plot(times, f0, color='k')
    plt.fill_between(times, 35 / 36 * f0, 36 / 35 * f0, color='lightgrey')

    for chunk in range(len(chunk_labels)):
        # print(chunk)
        alpha = 1.0
        if chunk_labels[chunk] == 0:
            col = 'w'
        if chunk_labels[chunk] == 1:
            col = 'w'  # 'r'
        if chunk_labels[chunk] == 2:
            if class_array[chunk] == 0:
                col = 'gold'

            if class_array[chunk] == 1:
                col = 'lightblue'#'tab:blue'#

            if class_array[chunk] == 2:
                col = 'red'#'red'
                alpha = 0.85

        plt.axvspan(chunk * chunk_size / sr, (chunk + 1) * chunk_size / sr, facecolor=col, alpha=alpha, zorder=-1)

        for value in ref_values[np.logical_and((ref_values < (18 / 17) * max_f), (ref_values > min_f * (17 / 18)))]:
            plt.axhline(value, color='k', linestyle=':', linewidth=0.5)

        plt.semilogy()
        ax = plt.gca()
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        plt.yticks(ref_values, ref_ticks)

        ax.set_xlim(left=-0.01)
        ax.set_ylim(top=(18 / 17) * max_f, bottom=min_f * (17 / 18))
        ax.set_xlabel('time [s]', weight='bold')
        ax.set_xticks(range(0, int(np.ceil(times[-1]) + 1)))

    figc = plt.gcf()
    figc.patch.set_facecolor('gainsboro')  #whitesmoke
    figc.patch.set_alpha(0.20)
    figc.tight_layout()
    figc.subplots_adjust(bottom=0.15)
    print('saving_plot!')
    print(path)
    #plt.savefig(os.path.join(path, 'pitch_plot.png'),dpi=300 )
    plt.savefig(os.path.join(path, 'pitch_plot.svg'),format = 'svg', dpi=300)

    return 'pitch_plot.svg'
