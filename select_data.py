import os
import argparse
import numpy as np
import pandas as pd
import re
import time
import random
import numpy
from scipy import stats
from utils import get_colnames_from_dict

# convolution_feature_cols = ['batchsize', 'matsize', 'kernelsize', 'channels_in', 'channels_out', 'strides', 'padding', 'activation_fct', 'use_bias', 'elements_matrix', 'elements_kernel']
# dense_feature_cols = ['batchsize', 'dim_input', 'dim_output', 'activation_fct']
# pooling_feature_cols = ['batchsize', 'matsize', 'channels_in', 'poolsize', 'strides', 'padding', 'elements_matrix']
# time_cols = ['hashkey', 'preprocess_time', 'execution_time', 'memcpy_time', 'retval_time', 'retval_half_time', 'memcpy_retval', 'memcpy_retval_half', 'sess_time', 'time_max', 'time_min', 'time_median', 'time_mean', 'time_trim_mean']


def read_flags():
    parser = argparse.ArgumentParser('Select Data Parser')
    # Benchmarks parameters
    parser.add_argument('--device', '-d', type=str, default='', help='select device')
    parser.add_argument('--operation', '-op', type=str, default='', help='select operation')
    # parser.add_argument('--filename', '-f', type=str, default=os.path.join(os.getcwd(), 'struct','lenet.csv'), help='csv file')
    #parser.add_argument('--num_of_layers', '-num', type=int, default=1, help='number of layers')
    # parser.add_argument('--start_from_csv', '-sfc', type=int, default=1, help='layer start')
    # parser.add_argument('--numbers_of_csv', '-nc', type=int, default=1, help='count of layers')
    # parser.add_argument('--one_layer', '-ol', action="store_true", default=False, help='just use one layer')


    # General parameters
    # parser.add_argument('--log_file', type=str, default='', help='Text file to store results')
    # parser.add_argument('--iter_warmup', type=int, default=5, help='Number of iterations for warm-up')    
    # parser.add_argument('--batch_size', '-b', type=int, default=1, help='Batch_size of data')
    # parser.add_argument('--iter_benchmark', type=int, default=10, help='Number of iterations for benchmark')    

    parser.add_argument('--output_path', type=str, default='', help='output path')
    parser.add_argument('--output_name', type=str, default='', help='output file name')

    args = parser.parse_args()

    
    args.output_path = os.path.join(os.getcwd(), args.device, 'golden_values')
    args.output_name = '%s_%s.csv'%(args.operation, args.device)

    if args.device == '' or args.operation == '':
        print('you need specific device and operation. ex: -d 1080ti -op convolution')
        exit()

    return args

def main():
    flags = read_flags()

    golden_file = os.path.join(os.getcwd(), 'all_golden_values', '%s_%s.csv' % (flags.operation, flags.device))
    print(golden_file)
    df_golden = pd.read_csv(golden_file)

    read_file_path_0 = os.path.join(os.getcwd(), '%s' % flags.device, 'golden_values_0', '%s_%s.csv' % (flags.operation, flags.device))
    print(read_file_path_0)
    df_0 = pd.read_csv(read_file_path_0)
    df_golden['preprocess_time_0'] = df_0['preprocess_time']
    df_golden['execution_time_0'] = df_0['execution_time']
    df_golden['memcpy_time_0'] = df_0['memcpy_time']
    df_golden['retval_time_0'] = df_0['retval_time']
    df_golden['retval_half_time_0'] = df_0['retval_half_time']
    df_golden['memcpy_retval_0'] = df_0['memcpy_retval']
    df_golden['memcpy_retval_half_0'] = df_0['memcpy_retval_half']
    df_0['sess_time'] = df_0['preprocess_time'] + df_0['execution_time'] + df_0['memcpy_retval_half']
    df_golden['sess_time_0'] = df_0['sess_time']

    read_file_path_1 = os.path.join(os.getcwd(), '%s' % flags.device, 'golden_values_1', '%s_%s.csv' % (flags.operation, flags.device))
    print(read_file_path_1)
    df_1 = pd.read_csv(read_file_path_1)
    df_golden['preprocess_time_1'] = df_1['preprocess_time']
    df_golden['execution_time_1'] = df_1['execution_time']
    df_golden['memcpy_time_1'] = df_1['memcpy_time']
    df_golden['retval_time_1'] = df_1['retval_time']
    df_golden['retval_half_time_1'] = df_1['retval_half_time']
    df_golden['memcpy_retval_1'] = df_1['memcpy_retval']
    df_golden['memcpy_retval_half_1'] = df_1['memcpy_retval_half']
    df_1['sess_time'] = df_1['preprocess_time'] + df_1['execution_time'] + df_1['memcpy_retval_half']
    df_golden['sess_time_1'] = df_1['sess_time']

    read_file_path_2 = os.path.join(os.getcwd(), '%s' % flags.device, 'golden_values_2', '%s_%s.csv' % (flags.operation, flags.device))
    print(read_file_path_2)
    df_2 = pd.read_csv(read_file_path_2)
    df_golden['preprocess_time_2'] = df_2['preprocess_time']
    df_golden['execution_time_2'] = df_2['execution_time']
    df_golden['memcpy_time_2'] = df_2['memcpy_time']
    df_golden['retval_time_2'] = df_2['retval_time']
    df_golden['retval_half_time_2'] = df_2['retval_half_time']
    df_golden['memcpy_retval_2'] = df_2['memcpy_retval']
    df_golden['memcpy_retval_half_2'] = df_2['memcpy_retval_half']
    df_2['sess_time'] = df_2['preprocess_time'] + df_2['execution_time'] + df_2['memcpy_retval_half']
    df_golden['sess_time_2'] = df_2['sess_time']


    

    # df_ = pd.read_csv(read_file_path_0, usecols=feature_cols)
    # print(df_.shape)
    print(df_golden)

    df_min = pd.DataFrame({"0":abs(df_golden['sess_time_0'] - df_golden['time_mean']), 
                   "1":abs(df_golden['sess_time_1'] - df_golden['time_mean']), 
                   "2":abs(df_golden['sess_time_2'] - df_golden['time_mean'])})
    # print(df_min)
    # print(df_min.idxmin(axis=1))
    df_golden['smallest_idx'] = df_min.idxmin(axis=1)

    df_golden['smallest_preprocess_time_idx_name'] = 'preprocess_time_' + df_golden['smallest_idx']
    df_golden['smallest_execution_time_idx_name'] = 'execution_time_' + df_golden['smallest_idx']
    df_golden['smallest_memcpy_time_idx_name'] = 'memcpy_time_' + df_golden['smallest_idx']
    df_golden['smallest_retval_time_idx_name'] = 'retval_time_' + df_golden['smallest_idx']
    df_golden['smallest_retval_half_time_idx_name'] = 'retval_half_time_' + df_golden['smallest_idx']
    df_golden['smallest_memcpy_retval_idx_name'] = 'memcpy_retval_' + df_golden['smallest_idx']
    df_golden['smallest_memcpy_retval_half_idx_name'] = 'memcpy_retval_half_' + df_golden['smallest_idx']
    df_golden['smallest_sess_time_idx_name'] = 'sess_time_' + df_golden['smallest_idx']
    # print(df_golden['smallest_sess_time_idx_name'])

    # df_golden.assign(sess_time=df_golden.lookup(df_golden.index, df_golden['smallest_sess_time_idx_name']))
    df_golden['preprocess_time'] = df_golden.lookup(df_golden.index, df_golden['smallest_preprocess_time_idx_name'])
    df_golden['execution_time'] = df_golden.lookup(df_golden.index, df_golden['smallest_execution_time_idx_name'])
    df_golden['memcpy_time'] = df_golden.lookup(df_golden.index, df_golden['smallest_memcpy_time_idx_name'])
    df_golden['retval_time'] = df_golden.lookup(df_golden.index, df_golden['smallest_retval_time_idx_name'])
    df_golden['retval_half_time'] = df_golden.lookup(df_golden.index, df_golden['smallest_retval_half_time_idx_name'])
    df_golden['memcpy_retval'] = df_golden.lookup(df_golden.index, df_golden['smallest_memcpy_retval_idx_name'])
    df_golden['memcpy_retval_half'] = df_golden.lookup(df_golden.index, df_golden['smallest_memcpy_retval_half_idx_name'])
    df_golden['sess_time'] = df_golden.lookup(df_golden.index, df_golden['smallest_sess_time_idx_name'])

    if flags.operation == 'convolution':
        df_golden['elements_matrix'] = df_2['elements_matrix']
        df_golden['elements_kernel'] = df_2['elements_kernel']
    elif flags.operation == 'pooling':
        df_golden['elements_matrix'] = df_2['elements_matrix']

    df_golden['hashkey'] = df_2['hashkey']
    cols_dict = get_colnames_from_dict()
    cols_ = cols_dict[flags.operation]
    cols_.extend(cols_dict['hash'])
    cols_.extend(cols_dict['time'])
    cols_.extend(cols_dict['profile'])
    # print(cols_)

    # exit()


    if not os.path.isdir(flags.output_path):
        os.makedirs(flags.output_path)
    print(os.path.join(flags.output_path, flags.output_name))
    df_golden.to_csv(os.path.join(flags.output_path, flags.output_name), columns = cols_, index=False)
    


if __name__ == '__main__':
    main()

