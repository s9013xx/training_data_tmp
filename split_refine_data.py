import os
import sys
import argparse
import numpy as np
import pandas as pd
import numpy
import glob

parser = argparse.ArgumentParser('Split Data Parser')
# Benchmarks parameters
parser.add_argument('--convolution', action="store_true", default=False, help='Benchmark convolution layer')
parser.add_argument('--dense', action="store_true", default=False, help='Benchmark fully connection layer')
parser.add_argument('--pooling', action="store_true", default=False, help='Benchmark pooling layer')
# General parameters
parser.add_argument('--device', type=str, default='', help='Device name as appearing in logfile')
# Check is inference or training
args = parser.parse_args()

if args.convolution:
    operation = 'convolution'
    golden_values_col = ['batchsize', 'matsize', 'kernelsize', 'channels_in', 'channels_out', 'strides', 'padding', 'activation_fct', 'use_bias', 'time_max', 'time_min', 'time_median', 'time_mean', 'time_trim_mean', 'preprocess_time', 'execution_time', 'memcpy_time', 'retval_time', 'mem_ret', 'retval_half_time', 'sess_time', 'elements_matrix', 'elements_kernel']
elif args.dense:
    operation = 'dense'
    golden_values_col = ['batchsize', 'dim_input', 'dim_output', 'activation_fct', 'time_max', 'time_min', 'time_median', 'time_mean', 'time_trim_mean', 'preprocess_time', 'execution_time', 'memcpy_time', 'retval_time', 'mem_ret', 'retval_half_time', 'sess_time']
elif args.pooling:
    operation = 'pooling'
    golden_values_col = ['batchsize', 'matsize', 'channels_in', 'poolsize', 'padding', 'strides', 'time_max', 'time_min', 'time_median', 'time_mean', 'time_trim_mean', 'preprocess_time', 'execution_time', 'memcpy_time', 'retval_time', 'mem_ret', 'retval_half_time', 'sess_time', 'elements_matrix']
else:
    print('you should use specific which operation data you want to split, ex: --convolution')
    exit()


if args.device == '':
    print('you should use --device parameter to specify collect data for which device, ex: --device 2080ti')
    exit()

# This funcion is filter for non-zero value
def data_filter(df_ori_data):
    # Time invalid filter
    df_data = df_ori_data.dropna()
    # re-index for split data
    df_data = df_data.reset_index(drop=True)

    return df_data

# This funcion is divide total data to three parts (train, validate, test)
def data_divider(df_ori_data, start_index, end_index):
    # Divide total data to train, validation, test part
    df_data = df_ori_data.loc[start_index:end_index, :]

    return df_data

def main():
    # step1 get the dataset 
    
    # read_file_path = os.path.join('goldan_values', '%s_goldan_values_%s_*.csv' % (operation, args.device))
    read_file_path = os.path.join(os.getcwd(), 'all_golden_values', '%s_%s.csv' % (operation, args.device))

    df_ori = pd.read_csv(read_file_path, index_col=None)
    print(df_ori.shape)
    df_ori = data_filter(df_ori)
    print(df_ori.shape)

    if args.convolution:
        df_ori['elements_matrix'] = df_ori['matsize'] ** 2
        df_ori['elements_kernel'] = df_ori['kernelsize'] **2
    elif args.pooling:
        df_ori['elements_matrix'] = df_ori['matsize'] ** 2

    df_test_data_20000 = data_divider(df_ori, 0, 19999)
    df_train_data_80000 = data_divider(df_ori, 20000, 99999)
    
    store_data_path = os.path.join(os.getcwd(), 'data/%s/%s' % (args.device, operation))
    if not os.path.exists(store_data_path):
        os.makedirs(store_data_path)

    print(os.path.join(store_data_path, 'test.csv'))
    print(os.path.join(store_data_path, 'train.csv'))
    df_test_data_20000.to_csv(os.path.join(store_data_path, 'test.csv'), index=False)
    df_train_data_80000.to_csv(os.path.join(store_data_path, 'train.csv'), index=False)



if __name__ == '__main__':
    main()
