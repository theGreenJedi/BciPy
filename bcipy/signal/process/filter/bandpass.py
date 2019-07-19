import os

from scipy.signal import butter, sosfilt
import logging
import numpy as np

log = logging.getLogger(__name__)


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y


def text_filter(input_seq, filt=None, fs=256, k=2, filter_location=None):
    """
    :param input_seq: Input sequence to be filtered. Expected dimensions are 16xT
    :param filt: Input for using a specific filter. If left empty, according to
    :fs a pre-designed filter is going to be used. Filters are pre-designed for fs = 256,300 or 1024 Hz.
    :param fs: Sampling frequency of the hardware.
    :param k: downsampling order
    :param filter_location: Path to filters.txt, If left empty, filters.txt is assumed to be next to this sig_pro.py

    :return: output sequence that is filtered and downsampled input. Filter delay is compensated. Dimensions are 16xT/k

    256Hz
        - 1.75Hz to 45Hz
        - 60Hz -64dB Gain

    300Hz
        - 1.84Hz to 45Hz
        - 60Hz -84dB Gain

    1024Hz
        - 1.75Hz to 45Hz
        - 60Hz -64dB Gain

    """

    # If filter location is not provided, assume it is next to sig_pro.py file.
    if not filter_location:
        filter_location = os.path.dirname(
            os.path.abspath(__file__)) + '/resources/filters.txt'

    # Try to open the filters.txt file
    try:
        with open(filter_location, 'r') as text_file:
            dict_of_filters = eval(text_file.readline())
    except Exception as e:
        log.error(
            'filters.txt cannot be found in path that is passed:',
            filter_location)
        raise e

    # Try to get the required filter from the text file.
    try:
        filt = dict_of_filters[fs]
    except Exception as e:
        log.error(
            'filters.txt does not have a filter with sampling frequency provided.')
        raise e

    # Precision correction
    filt = np.array(filt)
    filt = filt - np.sum(filt) / filt.size

    # Initialize output sequence
    output_seq = [[]]

    # Convolution per channel
    for z in range(len(input_seq)):
        temp = np.convolve(input_seq[z][:], filt)
        # Filter off-set compensation
        temp = temp[int(np.ceil(len(filt) / 2.)) - 1:]
        # Downsampling
        output_seq.append(temp[::k])

    return np.array(output_seq[1:])
