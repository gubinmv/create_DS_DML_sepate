"""
Project: create_DS_DML_sepate
Source: https://github.com/gubinmv/create_DS_DML_sepate.git
Author: Maxim Gubin
"""
import librosa
import numpy as np
from numpy.lib import stride_tricks

import scipy
import scipy.io.wavfile
import soundfile as sf
import os, fnmatch

import tensorflow as tf

from project_param import _param

def find_files(directory, pattern=['*.wav', '*.WAV']):
    '''find files in the directory'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern[0]):
            files.append(os.path.join(root, filename))

    return files

def downsampling(source_files):
    '''Convert to new sampling rate'''

    new_rate = 8000

    for file in source_files:
            print("downsampling file = ", file)

            data, samplerate = sf.read(file)
            sf.write(file, data, samplerate, subtype='PCM_16')


            sampling_rate, audio = scipy.io.wavfile.read(file)
            if (len(audio.shape)==2):
                audio = audio.sum(axis=1) // 2
            number_of_samples = round(len(audio) * float(new_rate) / sampling_rate)
            audio = scipy.signal.resample(audio, number_of_samples)
            audio = audio.astype(dtype = np.int16)
            scipy.io.wavfile.write(file, new_rate, audio)

def norm_audio(source_files):
        '''Normalize the audio files before training'''

        for file in source_files:
            audio, sr = librosa.load(file, sr=8000)
##            sr, audio = scipy.io.wavfile.read(file)
            div_fac = 1 / np.max(np.abs(audio)) / 3.0
            audio = audio * div_fac
##            scipy.io.wavfile.write(file, sr, audio)
            sf.write(file, audio, sr)
            print("normalization file = ", file)


def remove_silent(audio):
    """ remuve silent from wav-data """

    trimed_audio = []
    indices = librosa.effects.split(audio, hop_length=_param.window, top_db=20)

    for index in indices:
        trimed_audio.extend(audio[index[0]: index[1]])
    return np.array(trimed_audio)

def get_wav_files(list_file):
    """ Get wav-data from lest files """

    path_from = _param.path_source_DataSet
    wave_data = []
    for i in range(len(list_file)):
        path_file = path_from + str(list_file['file_name'].values[i])
        wave_data_hlp, sr = librosa.load(path_file, sr=8000)
        wave_data_hlp = remove_silent(wave_data_hlp)
        wave_data.append(wave_data_hlp)

    return wave_data

def my_stft(sig, frameSize=_param.window, overlapFac=_param.k_overlap, window=np.hanning): #256
    """ short time fourier transform of audio signal """

    win = window(frameSize)
    hopSize = int(frameSize - int(overlapFac * frameSize))
    samples = np.array(sig, dtype='float64')
    cols = int(np.ceil((len(samples) - frameSize) / float(hopSize)) + 1)
    samples = np.append(samples, np.zeros(frameSize))
    frames = stride_tricks.as_strided(
        samples,
        shape=(cols, frameSize),
        strides=(samples.strides[0] * hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)

def my_stft_2(wave_data, frameSize=_param.window, overlapFac=_param.overlap):
    """ short time fourier transform of audio signal """

    f, t, in_stft = scipy.signal.stft(wave_data, fs=_param.fs, nperseg=frameSize, noverlap = overlapFac, window='hann')
    in_stft = np.transpose(in_stft)

    return in_stft*100

def get_stft_samples(wav_file):
    '''
         Get spectrogramm from wav-file.
         input:
               wav_file: data of wav-file
         output:
               spectrogramms
    '''

    img_rows, img_cols = _param.img_rows, _param.img_cols

    in_stft = my_stft_2(wave_data=wav_file)

    in_stft_amp = np.maximum(np.abs(in_stft), 1e-5)
    in_data = in_stft_amp
    in_data = np.transpose(in_data)

    num_samples = in_data.shape[1]-img_cols

    sftt_frame = np.array([in_data[:, i:i+img_cols] for i in range(0, num_samples, 1)])

    return sftt_frame

def save_wav_file(file_name, wav_data):
    '''
         Save wav-file.
         input:
               file_name: path and file name
               wav_data:  data of audio
    '''


    scipy.io.wavfile.write(file_name, 8000, wav_data)

def my_stoi(wav_clean, wav_predict):
    '''
         Calculate the STOI indicator between the two audios.
         The larger the value, the better the separation.
         input:
               wav_predict: Generated audio
               wav_clean:  Ground Truth audio
         output:
               SNR value
    '''

    k = 15 # one-third octave band
    num_band_in_octave = len(wav_clean) // k

    a = np.array([np.sqrt(sum(wav_clean[num_band_in_octave * i:num_band_in_octave + num_band_in_octave * i]**2)) for i in range(0, k, 1)])
    a = np.append(a, np.sqrt(sum(wav_clean[num_band_in_octave * k:]**2)))

    return a

def get_SDR(wav_clean, wav_predict, fs = _param.fs, window = _param.window, overlap = _param.overlap):
    '''
         Calculate the SNR indicator between the two audios.
         The larger the value, the better the separation.
         input:
               wav_predict: Generated audio
               wav_clean:  Ground Truth audio
         output:
               SNR value
    '''

    f_1, t_1, ftt_clean_target = scipy.signal.stft(wav_clean, fs=fs, nperseg=window, noverlap = overlap, window='hann')
    f_2, t_2, ftt_predict_target = scipy.signal.stft(wav_predict, fs=fs, nperseg=window, noverlap = overlap, window='hann')

    ftt_clean_target = np.transpose(ftt_clean_target.real)
    ftt_predict_target = np.transpose(ftt_predict_target.real)

    num_segments = min(len(ftt_clean_target),len(ftt_predict_target))-10
    SDR_l2 = np.linalg.norm(ftt_clean_target[:num_segments], ord=2)**2 / (np.linalg.norm((ftt_clean_target[:num_segments] - ftt_predict_target[:num_segments]), ord=2)**2)
    SDR_l2 = 10*np.log10(SDR_l2)

    return SDR_l2

def SI_SNR(wav_predict, wav_clean):
    '''
         Calculate the SI_SNR indicator between the two audios.
         The larger the value, the better the separation.
         input:
               wav_predict: Generated audio
               wav_clean:  Ground Truth audio
         output:
               SNR value
    '''

    wav_predict = wav_predict - np.mean(wav_predict)
    wav_clean = wav_clean - np.mean(wav_clean)
    s_target = sum(np.multiply(wav_predict, wav_clean))*wav_clean/np.power(np.linalg.norm(wav_clean, ord=2), 2)
    e_noise = wav_predict - s_target

    return 20*np.log10(np.linalg.norm(s_target, ord=2)/np.linalg.norm(e_noise, ord=2))