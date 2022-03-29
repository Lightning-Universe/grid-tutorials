import os
import time
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data_dir', type=str)
args = parser.parse_args()
dir_list = os.listdir(args.data_dir)
time.sleep(60)
print(dir_list)
