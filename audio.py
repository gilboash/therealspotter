import sounddevice as sd
import numpy as np

def beep(freq=1000, dur=0.05):
    t = np.linspace(0, dur, int(44100 * dur))
    wave = np.sin(2 * np.pi * freq * t)
    sd.play(wave, 44100)
    sd.wait()

def ok():
    beep(800)

def wrong():
    beep(200, 0.15)