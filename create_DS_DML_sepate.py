"""
Project: create_DS_DML_sepate
Source: https://github.com/gubinmv/create_DS_DML_sepate.git
Author: Maxim Gubin
"""
import os
import numpy as np
import pandas

from project_param import _param
import project_function

from sklearn.utils import shuffle

def get_mix(noise, target):

    len_noise = len(noise)
    len_target = len(target)

    if (len_target > len_noise):
        k_ = 1 + len_target // len_noise
        noise = [noise] * k_
        noise = np.concatenate(noise, axis=0)
    else:
        k_ = 1 + len_noise // len_target
        target = [target] * k_
        target = np.concatenate(target, axis=0)

    tot_len = min(len(target), len(noise))

    mix = target[:tot_len] + noise[:tot_len]

    return target[:tot_len], noise[:tot_len], mix


def get_data_set(path_to_file_npz, list_noise, list_target, prefix):
    data_X = []
    data_Y = []

    fs = _param.fs
    window = _param.window
    overlap = _param.overlap
    step_wave = _param.step_wave

    img_rows, img_cols = _param.img_rows, _param.img_cols
    maxRazmer = _param.maxRazmer

    print("\n Loading wav files ...")

    list_wav_data_noise = project_function.get_wav_files(list_noise)
    list_wav_data_target = project_function.get_wav_files(list_target)

    noise = np.concatenate(list_wav_data_noise, axis=0)
    target = np.concatenate(list_wav_data_target, axis=0)

    target, noise, mix = get_mix(noise, target)

    noisePower = sum(noise**2)/len(noise)
    signalPower = sum(target**2)/len(target)

    snr_ = _param.snr_in * 0.1 / 2
    n_rate = np.sqrt(signalPower / noisePower)
    k_mix_noise = np.abs(n_rate / 10**snr_)
    print("\n k_mix_noise = ", k_mix_noise)
    noise = k_mix_noise * noise
    mix = target + noise

    noisePower = sum(noise**2)/len(noise)

    snr = 10*np.log10(signalPower/noisePower)
    print("\n SNR(dB) = ", snr)

    path_wav_out = _param.path_wav_out + prefix
    print("path as ", path_wav_out)

    if not os.path.exists(_param.path_wav_out):
        os.makedirs(_param.path_wav_out)

    if (model_name == 'Backward'):
        project_function.save_wav_file(path_wav_out + 'data_x.wav', mix)
        project_function.save_wav_file(path_wav_out + 'data_y.wav', noise)
        project_function.save_wav_file(path_wav_out + 'target.wav', target)

        stft_y = project_function.get_stft_samples(wav_file=noise)
        stft_x = project_function.get_stft_samples(wav_file=mix)

    elif (model_name == 'Forward'):
        project_function.save_wav_file(path_wav_out + 'data_x.wav', mix)
        project_function.save_wav_file(path_wav_out + 'data_y.wav', target)
        project_function.save_wav_file(path_wav_out + 'noise.wav', noise)

        stft_y = project_function.get_stft_samples(wav_file=target)
        stft_x = project_function.get_stft_samples(wav_file=mix)
    else:
        print('\n Stop! not method model \n')
        exit() #os._exit(0)


    x_train = np.array([])
    x_train = stft_x

    y_train = np.array([])
    y_train = stft_y[:, :, img_cols-1]

    print('\n shape baze')
    print("x_train.shape = ",x_train[0].shape)
    print("y_train.shape = ",y_train[0].shape)
    print('len train records x = ', len(x_train))
    print('len train records y = ', len(y_train))

    tot_len = min(len(x_train),len(y_train))
    y_train = y_train[:tot_len]
    x_train = x_train[:tot_len]

    #reshape tensor from conv_1D_inverse
    data_X = x_train.reshape(x_train.shape[0], img_rows, img_cols)
    data_Y = y_train.reshape(y_train.shape[0], img_rows, 1)

    print("save in npz-file ...")
    np.savez(path_to_file_npz, DataX=data_X, DataY=data_Y)

    return 0

def get_list_files(dataset_class, n_records, order = 1):

    list_files = pandas.DataFrame(columns=['file_name', 'class', 'length'])
    if (order == 1):
        n_records = n_records // len(dataset_class)
        for i in range(len(dataset_class)):
            list_file = df[df['class'].isin([dataset_class[i]])]
            list_files = pandas.concat([list_files, list_file[0:n_records]])
    else:
        list_files = df[df['class'].isin(dataset_class)].sample(n_records)[['file_name', 'class', 'length']]

    print("\n list fragments")
    print(list_files.groupby('class').count())

    return list_files

# start programm
print("\n Start programm creating spectrogramm.")

k_mix_noise = _param.k_mix_noise
model_name = _param.model_name

# open dataset
dataset_csv = _param.path_source_dataset_csv
df = pandas.read_csv(dataset_csv, delimiter=',')
pandas.set_option('display.max_rows', None)
print(df['class'].value_counts())

# get noise train
dataset_class = _param.class_noise
n_records = _param.n_records_noise
list_csv_noise_train = get_list_files(dataset_class = dataset_class, n_records = n_records, order = 0)

# get noise test
dataset_class = _param.class_noise_test
n_records = _param.n_records_noise_test
list_csv_noise_test = get_list_files(dataset_class = dataset_class, n_records = n_records, order = 0)

# get target train
dataset_class = _param.class_target
n_records = _param.n_records_target
list_csv_target_train = get_list_files(dataset_class = dataset_class, n_records = n_records)

# get target test
dataset_class = _param.class_target_test
n_records = _param.n_records_target_test
list_csv_target_test = get_list_files(dataset_class = dataset_class, n_records = n_records)

# create npz for train phase
print("\n Creating spectrogram for train")
path_to_file_npz = _param.path_to_file_train_npz
hlp = get_data_set(path_to_file_npz, list_csv_noise_train, list_csv_target_train, 'train_')

# create npz for test phase
print("\n Creating spectrogram for test")
path_to_file_npz = _param.path_to_file_test_npz
hlp = get_data_set(path_to_file_npz, list_csv_noise_test, list_csv_target_test, 'test_')

print("\n End programm creating spectrogramm.")
