import pandas as pd
import os
import argparse
from os.path import join as pjoin
import glob

def csv_to_txt_files(args):

    for type in ['valid', 'test', 'train']:
        for f in glob.glob(pjoin(args.raw_path, type + '.csv')):
            data_type = f.split('/')[-1].split('.')[0]

            output_path = os.path.join(args.save_path, data_type)
            os.makedirs(output_path, exist_ok=True)

            df = pd.read_csv(f)
            num_orig = len(df)
            #print(num_orig)
            num_gene = 0
            for index, row in df.iterrows():
                # 每行数据生成一个独立的txt文件
                txt_file_path = os.path.join(output_path, f'attack_{data_type}_{index}.txt')
                #print(txt_file_path)
                with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
                    txt_file.write(row[1].strip() + '\n\n@highlight\n\n' + row[0].strip() + '\n\n')
                    num_gene += 1

            if num_orig != num_gene:
                raise Exception(
                    "The %s dataset directory contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
                        output_path, num_gene, num_orig, num_gene))
            print("Successfully finished segmenting data %s to %s.\n" % (f, output_path))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-raw_path", default='../../line_data')
    parser.add_argument("-save_path", default='../../data/')

    args = parser.parse_args()
    csv_to_txt_files(args)
