import time
import argparse
parser = argparse.ArgumentParser("")
parser.add_argument("--dataset", type=str)
parser.add_argument("--itera", type=int)
parser.add_argument("--div", type=int)
args = parser.parse_args()
start_time = time.time()

itera = args.itera
# 调整位置后数据
f_1 = open(f"{args.dataset}_test.csv", 'w', encoding='utf-8')
with open(f'{args.dataset}_testall.csv', 'r', encoding='utf-8') as t:
    for ii, text in enumerate(t):
        text = text.split(',')
        text = ' '.join(text[1:])
        f_1.write(text)
t.close()
f_1.close()

import os

for i in range(1, int(itera) + 1):
    read_path = f"{args.dataset}_prelabel" + str(i) + ".csv"
    save_path = f"{args.dataset}_prelabel_" + str(i) + ".csv"

    f_3 = open(save_path, 'w', encoding='utf-8')
    with open(read_path, 'r', encoding='utf-8') as f:
        for ii, label1 in enumerate(f):
            label1 = int(label1)
            if label1 == 0:
                f_3.write("-1" + '\n')
            else:
                label1 = str(label1)
                f_3.write(label1 + '\n')
    f.close()
    f_3.close()

# 先将0改为-1，用0覆盖训练集
read_path1 = f'{args.dataset}_test0.csv'
m = 0
n = 0
with open(read_path1, 'r', encoding='utf-8') as f:
    for ii, text in enumerate(f):
        text = text.split(',')
        label1 = text[0]
        if label1 == '"1"':
            m += 1
        else:
            n += 1
print(m, n)
k = 0
j = m
for i in range(1, itera + 1):
    save_path = f"{args.dataset}_label" + str(i) + ".csv"
    f_2 = open(save_path, 'w', encoding='utf-8')
    read_path2 = f"{args.dataset}_prelabel_" + str(i) + ".csv"
    print('-----------')
    with open(read_path2, 'r', encoding='utf-8') as f:
        for ii, label in enumerate(f):
            label = int(label)
            if k + args.div <= m:
                if k <= ii < k + args.div:    #train1
                    pass
            elif m - k < args.div:
                if k <= ii < m or 0 <= ii < args.div - m + k:
                    pass

            if j + args.div <= m + n:
                if j <= ii < j+args.div:   #train1
                    pass
            elif m+n-j < args.div:
                if j <= ii < m+n or m <= ii < j + args.div - n:
                    pass
    if k+args.div <= m:
        print("train{}:{}-{} ".format(i, k, k + args.div))
        with open(read_path2, 'r', encoding='utf-8') as f:
            for ii, label in enumerate(f):
                if ii < m:
                    label = int(label)
                    if k <= ii < k+args.div:
                        if label == 1 or label == -1:
                            f_2.write("0" + '\n')
                    else:
                        label = str(label)
                        f_2.write(label + '\n')
    elif m - k < args.div:
        print("train{}:{}-{},{}-{}".format(i, k, m, 0, args.div-m+k))
        with open(read_path2, 'r', encoding='utf-8') as f:
            for ii, label in enumerate(f):
                if ii < m:
                    label = int(label)
                    if k <= ii < m or 0 <= ii < args.div-m+k:
                        if label == 1 or label == -1:
                            f_2.write("0" + '\n')
                    else:
                        label = str(label)
                        f_2.write(label + '\n')

    if j + args.div <= m + n:
        print("train{}:{}-{}".format(i, j, j + args.div))
        with open(read_path2, 'r', encoding='utf-8') as f:
            for ii, label in enumerate(f):
                if ii >= m:
                    label = int(label)
                    if j <= ii < j+args.div:
                        if label == 1 or label == -1:
                            f_2.write("0" + '\n')
                    else:
                        label = str(label)
                        f_2.write(label + '\n')
    elif m + n - j < args.div:
        print("train{}:{}-{},{}-{}".format(i, j, m + n, m, j + args.div - n))
        with open(read_path2, 'r', encoding='utf-8') as f:
            for ii, label in enumerate(f):
                if ii >= m:
                    label = int(label)
                    if j <= ii < m+n or m <= ii < j+args.div-n:
                        if label == 1 or label == -1:
                            f_2.write("0" + '\n')
                    else:
                        label = str(label)
                        f_2.write(label + '\n')
    k += args.div
    if k > m:
        k = k-m
    j += args.div
    if j > m+n:
        j = j-n

for i in range(1, itera + 1):
    os.remove(f"{args.dataset}_prelabel_" + str(i) + ".csv")

end_time = time.time()
run_time = end_time - start_time
with open('time.csv', 'a+', encoding='utf-8') as t:
    t.write(f'{args.dataset}, step4 runtime: ' + str(run_time) + '\n')

print('4cover down')