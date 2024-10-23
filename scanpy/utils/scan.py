import os

# Set environment variable before importing sounddevice. Value is not important.
os.environ["SD_ENABLE_ASIO"] = "1"

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

_TRIG_LEVEL_ = 1 #Trigger signal level

class SynchronizationError(ValueError):
    pass

def verify_input_types(inputs, types, **kwargs):
    """
    Make sure provided input types are valid

    :param inputs: List of input parameters
    :param types: List of correpsoning types
    :param kwargs:
    :return: None
    """

    for val, t in zip(inputs, types):
        if not isinstance(val, t):
            raise TypeError(f"Input {val} is not of type {t}")

def verify_synchronization(dwelltime, samplerate=48000):
    if not dwelltime >= 0:
        raise SynchronizationError(f"Sampling time {dwelltime} ms must be positive.")

    if (dwelltime/1000) % (1/samplerate) >= 1E-10:
        raise SynchronizationError(f"Sampling time {dwelltime} is incompatible with sampling rate {samplerate} Hz. It should be an integer multiple of {1/samplerate*1000} ms")

def pixels2time(n, dwelltime, samplerate=48000):
    """
    Return the number of sampling points given a number of pixels and dwelltime

    :param n: number of pixels
    :param dwelltime: the dwelltime in ms
    :param samplerate: The sampling rate in Hz. Default is 48kHz
    :return: the number of sampling points
    :rtype: int
    """

    verify_synchronization(dwelltime, samplerate)
    verify_input_types([n, samplerate], [int, int])

    t = n * dwelltime / 1000 * samplerate

    #if not abs(t - round(t, 0)) < 1E-15:
    #    raise ValueError(f"Incompatible pixels ({n}) and dwelltime ({dwelltime} ms) for samplerate ({samplerate} Hz). Resulting number of sampling points {t} is not an integer!")

    return int(t)

def trigger_signal(n, dwelltime, pixel_delay=0, duration=None, samplerate=48000):
    """
    Return a trigger signal

    :param n: number of triggers
    :param dwelltime: the dwell time between each trigger in ms
    :param pixel_delay: Delay/offset of the trigger in ms. Default is 0.
    :param duration: duration of the trigger in ms. Default is 0.
    :param samplerate: The sampling rate in Hz. Default is 48kHz.
    :return: trig
    :rtype: numpy.ndarray
    """
    if duration is None:
        duration = 1000/samplerate

    trig = np.zeros(pixels2time(n, dwelltime, samplerate))

    for i in range(n - 1):
        trigger_start = pixels2time(i+1, dwelltime, samplerate) + pixels2time(1, pixel_delay, samplerate)
        #trigger_start = int(((i + 1) * dwelltime + pixel_delay)/ 1000 * samplerate)
        #trigger_stop = int(trigger_start + duration/1000*samplerate)
        trigger_stop = trigger_start + pixels2time(1, duration, samplerate)

        trig[trigger_start:trigger_stop] = _TRIG_LEVEL_

    return trig
def pixel_signal(n, dwelltime, stepsize, samplerate=48000):
    """
    Return a pixel signal
    :param n: number of pixels
    :param dwelltime: the dwell time between each pixel in ms
    :param stepsize: stepsize in signal level between each pixel
    :param samplerate: The sampling rate in Hz. Default is 48kHz.
    :return: x
    :rtype: numpy.ndarray
    """

    x = np.zeros(pixels2time(n, dwelltime, samplerate)) - n*stepsize/2
    for i in range(n - 1):
        #x[int((i + 1) * dwelltime / 1000 * samplerate):] += stepsize
        x[pixels2time(i + 1,dwelltime, samplerate):] += stepsize

    return x

def line_signal(nx, ny, current_line, dwelltime, stepsize, samplerate=48000):
    """
    Return a line signal
    :param nx: number of pixels on the line
    :param ny: number of lines
    :param current_line: The current line number
    :param stepsize: The stepsize between lines
    :param samplerate: The sampling rate in Hz. Default is 48kHz.
    :return: y
    :rtype: numpy.ndarray
    """
    if current_line < 0 or current_line >= ny:
        raise ValueError(f"Line number {current_line} out of range for scan with {ny} lines")

    #return np.ones(int(nx * dwelltime / 1000 * samplerate))*stepsize*current_line - (ny-1)*stepsize/2
    return np.ones(pixels2time(nx, dwelltime, samplerate)) * stepsize * current_line - (ny - 1) * stepsize / 2


def scan_generator(nx, ny, dx, dy, dwelltime, samplerate=48000, **kwargs):
    """
    Generate a scan signal

    :param nx: Number of pixels in a line
    :param ny: Number of lines in the scan
    :param dx: The stepsize between pixels
    :param dy: The stepsize between lines
    :param dwelltime: The dwelltime at each pixel in ms
    :param samplerate: The sampling rate in Hz. Default is 48kHz.
    :param kwargs: Optional keyword arguments passed to the trigger function
    :return:
    """

    x = pixel_signal(nx, dwelltime, dx, samplerate=samplerate)
    trigger = trigger_signal(nx, dwelltime, samplerate=samplerate, **kwargs)

    for i in range(ny):
        y = line_signal(nx, ny, i, dwelltime, dy, samplerate=samplerate)
        yield (trigger, x, y)

def plot_signals(trigger, x, y=None):

    if y is None:
        fig, ax = plt.subplots()
    else:
        fig, (ax, ax2) = plt.subplots(nrows=2)
        ax2.plot(y, 'k', linewidth=0.15, label='y')

    ax.plot(x, 'b', linewidth=0.5, label='pixel')
    ax.set_ylabel('Pixel signal level')
    ax.set_xlabel('Time (s)')

    twinax = ax.twinx()
    twinax.plot(trigger, 'r', linewidth=0.25, label='trigger')
    twinax.set_ylabel('Trigger level')

    ax.legend()

def scan_signal(nx=256, ny=256, dx=None, dy=None, dwelltime=None, samplerate=48000, **kwargs):
    """
    Output scan signals

    :param nx: Number of pixels in a line
    :param ny: Number of lines in the scan
    :param dx: Stepsize between pixels
    :param dy: Stepsize between lines
    :param dwelltime: Dwell time between each pixel in ms
    :param samplerate: The sampling rate in Hz. Default is 48kHz.
    :param kwargs: Optional keyword arguments passed to the trigger function
    :return: None
    """

    if dx is None:
        dx = 1/nx

    if dy is None:
        dy = 1/ny

    if dwelltime is None:
        dwelltime = 1000/samplerate

    fig, ax = plt.subplots()
    for (trigger, x, y) in scan_generator(nx, ny, dx, dy, dwelltime, samplerate, **kwargs):
        plot_signals(trigger, x, y)
    plt.show()
