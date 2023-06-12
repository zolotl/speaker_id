import librosa
import numpy as np
import random

# TIME STRETCH
def stretch(data, rate=random.uniform(0.8, 1.2)):
    return librosa.effects.time_stretch(data, rate)
# SHIFT
def shift(data, shift_range=int(np.random.uniform(low=-1.5, high = 1.5)*1000)):
    return np.roll(data, shift_range)
# PITCH
def pitch(data, sampling_rate, n_steps=random.randint(-3, 3)):
    return librosa.effects.pitch_shift(data, sampling_rate, n_steps)