"""
Project: create_DS_DML_sepate
Source: https://github.com/gubinmv/create_DS_DML_sepate.git
Author: Maxim Gubin
"""

class _param(object):

    # wave parameters
    fs          = 8000

    #spectrogramm parameters
    window      = 256
    k_overlap   = 3/4
    overlap     = int(k_overlap * window)
    step_wave   = int(window - overlap)

    # size of spectrogram
    img_rows    = 1 + window // 2
    img_cols    = 8

    # size wave from spectrogram
    maxRazmer = (window - overlap) * (img_cols - 1) + window

    #home dir
    home_dir    = './'

    # source DataSet
    path_source_DataSet     = 'c:/!!My_DataSet3/dataset/'
    path_source_dataset_csv = 'c:/!!My_DataSet3/dataset.csv'

    # dataset parameters
    class_noise                 = ['bigtown-scene1']
    n_records_noise             = 100
    class_noise_test            = ['bigtown-scene1-test']
    n_records_noise_test        = 100

    class_target             = ['vtarget1', 'target2', 'target3']
    n_records_target         = 100
    class_target_test        = ['target1-test', 'target2-test', 'target3-test']
    n_records_target_test    = 100

    # ratio noise to target signal
    k_mix_noise             = 0.9
    snr_in                  = 0.0

    # dataset [x,y] in npz-files
    path_to_file_test_npz   = "./TestSet.npz"
    path_to_file_train_npz  = "./TrainSet.npz"

    # model trening
    model_name              = 'Backward'
    file_name_model         = 'model_conv1d-15skip'
    file_history            = 'history_conv1d-15skip'
    path_history            = './' + file_history + '.txt'
    path_model              = './' + file_name_model + '.hdf5'

    # params learning
    batch_size  = 64
    epochs      = 10

    # params CNN
    l2_lambda   = 1e-5

    # path of save parameters
    path_wav_out            = './Out/'
