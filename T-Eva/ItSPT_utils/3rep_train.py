# -*- coding: utf-8 -*-
# 这一段是实体概念扩展

# 将test文件换为dataset_testall文件, fewshot.py第404行换为dataset_prelabel_0.csv, 第539行换为去掉注释

import logging
import sys
import subprocess
import time
import os
import argparse
parser = argparse.ArgumentParser("")
parser.add_argument("--dataset", type=str)
parser.add_argument("--itera", type=int)
parser.add_argument("--div", type=int)
args = parser.parse_args()
start_time = time.time()

if __name__ == '__main__':

    itera = args.itera
    with open('output.txt', 'a+', encoding='utf-8') as t:
        t.write('train: ' + str(itera) + '\n')

    file1 = open(f"{args.dataset}_testall.csv", "r")
    file2 = open(f"./datasets/TextClassification/{args.dataset}/test.csv", "w")

    m = file1.read()
    n = file2.write(m)

    file1.close()
    file2.close()

    for i in range(1, itera + 1):
        file3 = open(f"{args.dataset}_train" + str(i) + ".csv", "r")
        file4 = open(f"./datasets/TextClassification/{args.dataset}/train.csv", "w")

        s = file3.read()
        w = file4.write(s)

        file3.close()
        file4.close()

        dataset = args.dataset
        div = int((args.div) / 2)
        cmd = "python fewshot1.py --result_file ./output_f1.txt --dataset " + str(dataset) + \
              " --template_id 0  --seed 100 --shot " + str(div) + " --verbalizer manual"
        # p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # p.communicate()
        subprocess.run(cmd, shell=True, check=True)

        file5 = open(f"{args.dataset}_prelabel_0.csv", "r")
        file6 = open(f"{args.dataset}_prelabel" + str(i) + ".csv", "w")

        a = file5.read()
        b = file6.write(a)

        file5.close()
        file6.close()
        print("train {}:preprocess down".format(i))

        time.sleep(3)
    
    end_time = time.time()
    run_time = end_time - start_time
    with open('time.csv', 'a+', encoding='utf-8') as t:
        t.write(f'{args.dataset}, step3 runtime: ' + str(run_time) + '\n')

print('3rep_train down')


