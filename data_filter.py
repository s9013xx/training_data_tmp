import os
import numpy as np
import pandas as pd
from termcolor import colored
from flags import read_data_filter_flags
from utils import get_colnames_from_dict


def main():
    flags = read_data_filter_flags()
    warn_tag = colored('[Warn] ', 'red', attrs=['blink']) 
    success_tag = colored('[Success] ', 'green')

    if not os.path.isdir(flags.output_path):
        os.makedirs(flags.output_path)
    output_sub_path = os.path.join(flags.output_path, flags.predition_layertype)
    if not os.path.isdir(output_sub_path):
        os.makedirs(output_sub_path)

    tmp_str = flags.predition_layertype + '_' + flags.device + '.csv'
    if not flags.output_filename:
        flags.output_filename = os.path.join(flags.output_path, tmp_str)
        print(warn_tag + 'Auto create file: ' + flags.output_filename)

    df = pd.read_csv(flags.input_filename)

    df['time_diff'] = abs(df['sess_time'] - df['time_mean'])/df['time_mean']
    if flags.predition_layertype == 'convolution':
        df = df[df.time_diff < 0.1]
    if flags.predition_layertype == 'dense':
        df = df[df.time_diff < 0.25]
    if flags.predition_layertype == 'pooling':
        df = df[df.time_diff < 0.2]
    # df = df.sort_values(by=['time_diff'], ascending=False)
    df = df.reset_index(drop=True)
    # print(df)
    # print('total raws:', len(df))
    # print('number:', int(.2*len(df))-1)
    test_df = df.loc[0:int(.2*len(df))-1, :]
    train_df = df.loc[int(.2*len(df)):len(df)-1, :]
    # print(test_df)
    # print(train_df)
    df.to_csv(flags.output_filename, index=False)
    test_df.to_csv(os.path.join(output_sub_path, 'test.csv'), index=False)
    train_df.to_csv(os.path.join(output_sub_path, 'train.csv'), index=False)
    exit()

    col_dict = get_colnames_from_dict()
    df_all = None
    df_struct   = pd.read_csv(flags.input_struct_csv)
    df_time     = pd.read_csv(flags.input_time_csv)
    df_timeline = pd.read_csv(flags.input_timeline_csv)

    #print(df_time)
    #print(df_timeline)
    df_all = df_struct.copy()
    for ct in col_dict['profile']:
        df_all[ct] = df_all.hashkey.map(df_timeline.set_index('hashkey')[ct])

    for ct in col_dict['time']:
        df_all[ct] = df_all.hashkey.map(df_time.set_index('hashkey')[ct])
    df_all.to_csv(flags.output_filename, index=False)
    #print(df_all)
    print(success_tag + 'Data is Combined!')

if __name__ == '__main__':
    main()

