# scanner
Scanning electron beams through audio-signals in python

This python tool allows control of the electron beam through passing signals through an audio card to the beam deflector coils. Central to this tool is the [python-sounddevice](https://python-sounddevice.readthedocs.io/en/0.5.1/index.html) python package that enables ASIO control.

This is currently a work in progress, and there are many possible avenues to explore in order to find the best and most user friendly implementation. The possible approaches are:

- "Playing" numpy arrays directly, either through premade .npz files or by generating numpy arrays in memory.
- using [`asyncio`](https://docs.python.org/3/library/asyncio.html) functionality.

The ASIO signals should both emit signals to the scan coils and to the trigger output. Since all must be synchronized to a samplingrate of 48 000 Hz (determined by the NanoMegas P1000 scan generator hardware) the code must take this into account to create synchronous signals.

## Playing numpy arrays
Playing numpy arrays can lead to problems for large scans. With a samplingrate of 48 000 Hz, there will be 48 000 numbers in each numpy array
