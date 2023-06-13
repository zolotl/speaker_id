import librosa
import numpy as np
import random

# TIME STRETCH
def stretch(data, rate_range = (0.8, 1.2)):
    rate = random.uniform(rate_range[0], rate_range[1])
    return librosa.effects.time_stretch(data, rate=rate)
# SHIFT
def shift(data, shift_range=int(np.random.uniform(low=-1.5, high = 1.5)*1000)):
    return np.roll(data, shift_range)
# PITCH
def pitch(data, sampling_rate, n_steps=(-3, 3)):
    n_steps = random.randint(n_steps[0], n_steps[1])
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=n_steps)