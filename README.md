Setup: run pip install -r requirements.txt, then python mso58b_harmonic_analyzer.py.

The window has four tabs:

1. Scope connection: choose how to connect:
    Ethernet, direct: needs no extra drivers. On the scope, turn on Utility > I/O > Socket Server (port 4000, protocol “None”).
    Ethernet through VISA, or USB: for USB on Windows, install TekVISA or NI-VISA first. A Scan button finds the scope.
    Simulated scope: lets you try the app with no hardware.
2. Acquire data:
    Channels default to CH1–CH3 for phase A/B/C current and CH6–CH8 for voltage. You can change the mapping, and each channel has a scale factor in case a probe or transducer doesn’t read directly in amps or volts.
    You can take a single triggered capture or grab what’s already on screen. The app can set the sample rate and record length for you; the default is 2.5 MS/s for 0.5 s.
    It warns you if a current channel isn’t reading in amps, and shows a preview of the waveforms.
    Captures can be saved as CSV (which your MATLAB script can also open) or reloaded later. Your existing Tektronix/CDAQ CSV exports load too.
3. Harmonic analysis: the same settings as your script, with a coloured PASS/FAIL banner and per-phase current and voltage tables. Failing harmonics are highlighted in the harmonics table. “Save Excel report” writes the same five sheets, with the plots embedded and a PNG folder alongside.
4. Plots: your script’s six plots, each with zoom and pan.


PySide6>=6.5

numpy>=1.23

matplotlib>=3.6

openpyxl>=3.1

pillow>=9.0

pyvisa>=1.13

pyvisa-py>=0.7

