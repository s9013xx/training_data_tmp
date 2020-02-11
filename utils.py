import os
import sys

def get_support_devices():
    dict_ = {
        '1080ti': 'gpu',
    }
    return dict_

def get_support_layers():
    return ['convolution', 'dense', 'pooling']

def get_colnames(typename):
    if typename == 'convolution':
        return get_cov_colnames()
    elif typename == 'dense':
        return get_dense_colnames()
    elif typename == 'pooling':
        return get_pool_colnames()
    else:
        print("This type of layer is not support!")
        return

def get_hash_colnames():
    return ['hashkey']

def get_cov_colnames():
    return ['batchsize', 'matsize', 'kernelsize', 'channels_in', 'channels_out', 'strides', 'padding', 'activation_fct', 'use_bias', 'elements_matrix', 'elements_kernel']

def get_dense_colnames():
    return ['batchsize', 'dim_input', 'dim_output', 'activation_fct']

def get_pool_colnames():
    return ['batchsize', 'matsize', 'channels_in', 'poolsize', 'strides', 'padding', 'elements_matrix']

def get_time_colnames():
    return ['time_max', 'time_min', 'time_median', 'time_mean', 'time_trim_mean']

def get_profile_colnames():
    return ['preprocess_time', 'execution_time', 'memcpy_time', 'retval_time', 'retval_half_time', 'memcpy_retval', 'memcpy_retval_half', 'sess_time']#, 'elements_matrix', 'elements_kernel']

def get_colnames_from_dict():
    conv_colnames  = get_cov_colnames()
    dense_colnames = get_dense_colnames()
    pool_colnames  = get_pool_colnames()
    time_colnames  = get_time_colnames()
    profile_colnames = get_profile_colnames()
    cols_dict = {
        'convolution': conv_colnames,
        'dense': dense_colnames,
        'pooling': pool_colnames,
        'profile': profile_colnames,
        'time': time_colnames,
        'hash': get_hash_colnames()
    }
    return cols_dict