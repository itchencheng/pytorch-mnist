#coding:utf-8

import pandas as pd
import numpy as np


def one_to_some(dirname, file, is_labled):
    dataframe = pd.read_csv(file)
    data_size = dataframe.shape[0]

    fo = open(dirname+'info.txt', 'w')
    
    for i in range(data_size):
        print("%d/%d" %(i, data_size))
        tmp = np.array(dataframe.iloc[i])
        raw_name = '%d.raw' %(i)
    
        if (is_labled):
            fo.write("%d,%s\n" %(tmp[0], raw_name))
        else:
            fo.write("%s\n" %(raw_name))

        if (is_labled):
            tmp = np.float32(tmp[1:])
        else:
            tmp = np.float32(tmp)
        
        tmp.tofile(dirname+raw_name)
    print('Generated %s' %(file))

    fo.close()


def main():
    dataset_dir = '/home/chen/dataset/kaggle/digit-recognizer/'
    train_file = dataset_dir + 'train.csv'
    test_file  = dataset_dir + 'test.csv'

    one_to_some(dataset_dir + 'train/', train_file, True)
    one_to_some(dataset_dir + 'test/', test_file, False)

    print('Generated!')


if __name__ == "__main__":
    main()