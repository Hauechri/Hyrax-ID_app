from . import spectral, waveform, phase, cepstral, wavelet  # noqa: F401  (registers representations)
from .registry import register, get, create, available
from .compose import ChannelSpec, ImageSettings, compose, NORM_MODES
