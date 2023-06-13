import numpy as np

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


from preprocessing import train_data


def train_gmms(train_data, n_components=16, normalize=True):
  gmms = {}
  for speaker_id, samples in train_data.items():
    X = np.vstack([mfccs for mfccs, _ in samples])

    # Normalize MFCC features
    if normalize:
      scaler = StandardScaler()
      X = scaler.fit_transform(X)

    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X)
    gmms[speaker_id] = (gmm, scaler if normalize else None)

  return gmms

# Train the GMMs with the updated settings
gmms = train_gmms(train_data, n_components=16)
