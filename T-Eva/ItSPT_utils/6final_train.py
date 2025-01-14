# -*- coding: utf-8 -*-
# 这一段是实体概念扩展
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

import shutil
from glob import glob


def mycopyfile(srcfile, dstpath):  # 复制函数
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(srcfile)  # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)  # 创建路径
        shutil.copy(srcfile, dstpath + fname)  # 复制文件
        print("copy %s -> %s" % (srcfile, dstpath + fname))


if __name__ == '__main__':
    start_time = time.time()

    itera = args.itera
    with open('output.txt', 'a+', encoding='utf-8') as t:
        t.write('itera: ' + str(itera) + '\n')

    file1 = open("train_all.csv", "r")
    file2 = open(f"./datasets/TextClassification/{args.dataset}/train.csv", "w")

    s = file1.read()
    w = file2.write(s)

    file1.close()
    file2.close()

    # os.remove(f"ckpts/{args.dataset}_tem0.ckpt")


    for i in {100, 110, 120, 130, 140}:
        # src_dir = f'/home/ubuntu/user_wsq/ItSPT_human-chatgpt_model/{args.dataset}_tem0.ckpt'
        # dst_dir = '/home/ubuntu/user_wsq/ItSPT_human-chatgpt/ckpts/'  # 目的路径记得加斜杠
        # src_file_list = glob(src_dir + '*')  # glob获得路径下所有文件，可根据需要修改
        # for srcfile in src_file_list:
        #     mycopyfile(srcfile, dst_dir)
        dataset = args.dataset

        cmd = "python fewshot1.py --result_file ./output_f1.txt --dataset " + str(dataset) + \
              " --template_id 0  --seed  " + str(i) + " --shot 20 --verbalizer manual"
        # p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # p.communicate()
        subprocess.run(cmd, shell=True, check=True)

        print("train seed {}:preprocess down".format(i))

        # os.remove(f"ckpts/{args.dataset}_tem0.ckpt")

        time.sleep(3)

    end_time = time.time()
    run_time = (end_time - start_time) / 5 * 3

    for i in range(1, itera + 1):
        os.remove(f"{args.dataset}_label" + str(i) + ".csv")
    for i in range(0, itera + 1):
        os.remove(f"{args.dataset}_prelabel" + str(i) + ".csv")
    for i in range(1, itera + 1):
        os.remove(f"{args.dataset}_train" + str(i) + ".csv")
    os.remove("result.csv")
    os.remove("result1.csv")
    # os.remove("train_all.csv")
    os.remove(f"{args.dataset}_prelabel_0.csv")
    os.remove(f"{args.dataset}_test.csv")
    os.remove(f"{args.dataset}_test0.csv")
    os.remove(f"{args.dataset}_testall.csv")

    os.remove(f"ckpts/{args.dataset}_tem0.ckpt")

    with open('time.csv', 'a+', encoding='utf-8') as t:
        t.write(f'{args.dataset}, step6 runtime: ' + str(run_time) + '\n\n')

