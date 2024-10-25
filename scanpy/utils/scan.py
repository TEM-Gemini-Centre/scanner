import os

# Set environment variable before importing sounddevice. Value is not important.
os.environ["SD_ENABLE_ASIO"] = "1"

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from tabulate import tabulate

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

def time2index(time, samplerate):
    """
    Return the index at time `time` for sample rate `samplerate`

    :param time: The time in ms
    :type time: [int, float]
    :param samplerate: The sampling rate in Hz
    :type samplerate: [int, float]
    :return: index
    :rtype: float
    """

    index = (time / 1000) * samplerate

    return index

class ScanGenerator(object):
    """
    An object for generating scan signals
    """

    supported_pixel_signals=['sawtooth']
    _SIGNAL_SAFETY_CUTOFF_ = 0.75

    def __init__(self, samplerate=48000, nx=1, ny=1, dx=1, dy=1, dwelltime=None, flyback_delay=0., trigger_delay=0., trigger_duration=0., trigger_level = 1.):
        if dwelltime is None:
            dwelltime = 1000 / samplerate

        dwelltime=float(dwelltime)
        flyback_delay = float(flyback_delay)
        trigger_delay=float(trigger_delay)
        trigger_duration=float(trigger_duration)
        trigger_level=float(trigger_level)
        verify_input_types([samplerate, nx, ny, dx, dy, dwelltime, flyback_delay, trigger_delay, trigger_duration, trigger_level], [int, int, int, int, int, float, float, float, float, float])

        self.samplerate = samplerate
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.dwelltime = dwelltime
        self.flyback_delay = flyback_delay
        self.trigger_delay = trigger_delay
        self.trigger_duration = trigger_duration
        self.trigger_level = trigger_level


    def __repr__(self):
        return f'{self.__class__.__name__}({self.samplerate!r}, {self.nx!r}, {self.ny!r}, {self.dx!r}, {self.dwelltime!r}, {self.flyback_delay!r}, {self.trigger_delay!r}, {self.trigger_duration!r}, {self.trigger_level!r})'

    def __str__(self):
        table = tabulate([
            ['Samplerate', self.samplerate, 'Hz'],
            ['Nx', self.nx, 'px'],
            ['Ny', self.ny, 'px'],
            ['Dx', self.dx, 'arb.'],
            ['Dy', self.dy, 'arb.'],
            ['Dwelltime', self.dwelltime, 'ms'],
            ['FlybackDelay', self.flyback_delay, 'ms'],
            ['TriggerDelay', self.trigger_delay, 'ms'],
            ['TriggerDuration', self.trigger_duration, 'ms'],
            ['TriggerLevel', self.trigger_level, 'arb.'],
        ], headers=['Parameter', 'Value', 'Units'])
        return f'{self.__class__.__name__} with parameters:\n{table}\n'

    @property
    def linetime(self):
        return self.nx*self.dwelltime+self.flyback_delay

    @property
    def scantime(self):
        return self.ny*self.linetime

    @property
    def linelength(self):
        return self.time2index(self.linetime)

    def verify_synchronization(self, time):
        if not time >= 0:
            raise SynchronizationError(f"Sampling time {time} ms must be positive.")

        index = time2index(time, self.samplerate)
        if index - round(index, 0) >1E-10:
            raise SynchronizationError(f"Sampling time {time} ms is incompatible with sampling rate {self.samplerate} Hz. It should be an integer multiple of {1/self.samplerate*1000} ms")

    def time2index(self, time):
        """
        Return the index of the scan signal for a given time

        :param time: the time in ms
        :param samplerate: The sampling rate in Hz. Default is 48kHz
        :return: the index of the scan signal at time `time`
        :rtype: int
        """

        self.verify_synchronization(time)
        verify_input_types([time, self.samplerate], [float, int])

        return int(time2index(time, self.samplerate)) #use global function to calculate the index

    def trigger_signal(self):#, n, dwelltime, pixel_delay=0, duration=None, samplerate=48000):
        """
        Return a trigger signal

        :return: trig
        :rtype: numpy.ndarray
        """

        trig = np.zeros(self.linelength)

        trig[self.time2index(self.trigger_delay):self.time2index(self.trigger_delay) + self.time2index(self.trigger_duration)] = self.trigger_level #first trigger

        for i in range(self.nx - 1):
            trigger_start = self.time2index((i+1)*self.dwelltime) + self.time2index(self.trigger_delay)
            #trigger_start = int(((i + 1) * dwelltime + pixel_delay)/ 1000 * samplerate)
            #trigger_stop = int(trigger_start + duration/1000*samplerate)
            trigger_stop = trigger_start + self.time2index(self.trigger_duration)

            trig[trigger_start:trigger_stop] = self.trigger_level

        trig[trig>=self._SIGNAL_SAFETY_CUTOFF_] = self._SIGNAL_SAFETY_CUTOFF_
        return trig

    def sawtooth(self):
        """
        Return a sawtooth pixel signal

        :return: x
        :rtype: numpy.ndarray
        """

        x = np.zeros(self.linelength) - self.nx * self.dx / 2
        for i in range(self.nx - 1):
            x[self.time2index((i + 1) * self.dwelltime):] += self.dx
        if self.flyback_delay > 0:
            x[-self.time2index(self.flyback_delay):] = x[0]
        return x

    def pixel_signal(self, kind='sawtooth'):#, dwelltime, stepsize, samplerate=48000):
        """
        Return a pixel signal

        :return: x
        :rtype: numpy.ndarray
        """

        if kind == 'sawtooth':
            x = self.sawtooth()
        else:
            raise ValueError(f'Pixel signal {kind} not recognized. Supported pixel signals are {self.supported_pixel_signals}')

        x[abs(x)>=self._SIGNAL_SAFETY_CUTOFF_]=np.sign(x[abs(x)>=self._SIGNAL_SAFETY_CUTOFF_])*self._SIGNAL_SAFETY_CUTOFF_

    def line_signal(self, current_line):#nx, ny, current_line, dwelltime, stepsize, samplerate=48000):
        """
        Return a line signal

        :return: y
        :rtype: numpy.ndarray
        """
        if current_line < 0 or current_line >= self.ny:
            raise ValueError(f"Line number {current_line} out of range for scan with {self.ny} lines")

        y = np.ones(self.linelength) * self.dy * current_line - (self.ny - 1) * self.dy / 2
        if self.flyback_delay>0:
            y[-self.time2index(self.flyback_delay):] = y[0]+self.dy
        y[abs(y) >= self._SIGNAL_SAFETY_CUTOFF_] = np.sign(y[abs(y) >= self._SIGNAL_SAFETY_CUTOFF_]) * self._SIGNAL_SAFETY_CUTOFF_
        return y


    def scan_generator(self, snake=False):#nx, ny, dx, dy, dwelltime, samplerate=48000, **kwargs):
        """
        Generate a scan signal
        :return:
        """

        x = self.pixel_signal()
        trigger = self.trigger_signal()

        for i in range(self.ny):
            y = self.line_signal(i)
            if snake:
                if i%2:
                    yield (trigger, x[::-1], y)
                else:
                    yield (trigger, x, y)
            else:
                yield (trigger, x, y)

    def plot_signals(self, **kwargs):

        fig, ax = plt.subplots()
        axtwin= ax.twinx()

        generator = self.scan_generator(**kwargs)
        t = np.linspace(0, self.nx*self.dwelltime, len(self.pixel_signal()))
        for i, (trigger, x, y)  in enumerate(generator):
            ax.plot(t+(i*t[-1]), x, 'b', linewidth=1, label='x')
            axtwin.plot(t+(i*t[-1]), trigger, 'r', linewidth=0.25)
            ax.plot(t+(i*t[-1]), y, 'k', linewidth=2, label='y')


        ax.set_ylabel('Signal level')
        ax.set_xlabel('Time (ms)')
        #ax.legend()

        plt.show()