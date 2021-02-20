from pydub import AudioSegment
import numpy as np
import os
from scipy import signal
import sox
import random


def get_spectrogram(folder, filename):
    """
    Compute a spectrogram with Fourier transforms.

    Parameters
    ----------
    folder: string
            Path to the folder with song(s)
    filename: string
             Name of the file

    Returns
    -------
    times : ndarray
            Array of segment times.

    spect : ndarray
            Spectrogram of song.

    """
    #
    desired_rate = 11025
    # Read mp3 file from source path
    mp3 = AudioSegment.from_mp3(os.path.join(folder, filename))

    # Samples per second
    current_rate = mp3.frame_rate
    # Downsampling
    if current_rate != desired_rate:
        tfm = sox.Transformer()
        # change sample rate to desired_rate
        tfm.rate(samplerate=desired_rate)
        # create the output file.
        tfm.build(os.path.join(folder, filename), os.path.join(folder, "audio.mp3"))
        mp3 = AudioSegment.from_mp3(os.path.join(folder, "audio.mp3"))

        # delete the resampled file
        os.remove(os.path.join(folder, "audio.mp3"))
    # Extracting samples from an audio file
    mp3_samples = np.array(mp3.get_array_of_samples())
    rate = mp3.frame_rate

    # Stereo to mono
    if mp3.channels == 2:
        mp3_samples = mp3_samples.reshape((-1, 2))
        mp3_samples = np.mean(mp3_samples, axis=1)

    # FFT the signal and extract a spectrogram
    freqs, times, spect = signal.spectrogram(mp3_samples, fs=rate, window='hanning',
                                             nperseg=1024, noverlap=512,
                                             detrend=False)

    # Apply log transform since spectrogram function returns linear array
    spect = 10 * np.log10(spect, out=np.zeros_like(spect), where=(spect != 0))

    return spect, times


def split_spect(output_number, filename, spect, length=221):

    x = np.zeros((output_number, spect.shape[0], length))
    y = []
    for i in range(output_number):
        index = random.randint(0, spect.shape[1] - 1 - length)
        sample = spect[:, index:index+length]
        x[i] = sample
        y.append(filename)

    return x, y


def generate_datafile(audio_folder, output_folder,  output_number, length, output_file="data"):

    train_x = []
    train_y = []
    for filename in os.listdir(audio_folder):
        spect, _ = get_spectrogram(audio_folder, filename)
        x, y = split_spect(output_number, filename, spect,  length)
        train_x.extend(x)
        train_y.extend(y)

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    np.savez(os.path.sep.join([output_folder, output_file]), train_x=train_x, train_y=train_y)


def generate_spectdatafile(audio_folder, output_folder, output_file="spect"):
    train_x = []
    train_y = []

    for filename in os.listdir(audio_folder):
        spect, _ = get_spectrogram(audio_folder, filename)
        train_x.append(spect)
        train_y.append(filename)

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    np.savez(os.path.sep.join([output_folder, output_file]), train_x=train_x, train_y=train_y)







