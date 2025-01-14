import argparse
parser = argparse.ArgumentParser("")
parser.add_argument("--dataset", type=str)
parser.add_argument("--div", type=int)
args = parser.parse_args()

# 划分数据集{args.dataset}（args.div train，args.div val）
read_path = f'{args.dataset}_test0.csv'
m = 0
n = 0

with open(read_path, 'r', encoding='utf-8') as f:
    for ii, text in enumerate(f):
        text = text.split(',')
        label = text[0]
        if label == '"1"':
            m += 1
        else:
            n += 1
print(m, n)
k = 0
j = m
for i in range(1, 9):
    save_path = f"{args.dataset}_train" + str(i) + ".csv"
    f_w = open(save_path, 'w', encoding='utf-8')
    print('----------')
    with open(read_path, 'r', encoding='utf-8') as f:
        for ii, text in enumerate(f):
            text = text.split(',')
            label = text[0]
            text = ' '.join(text[1:]).replace('\n', '')
            if k + args.div <= m:
                if k <= ii < k + args.div:    #train1
                    f_w.write(label + ',' + text + '\n')
            elif m-k < args.div:
                if k <= ii < m or 0 <= ii < args.div - m + k:
                    f_w.write(label + ',' + text + '\n')

            if j + args.div <= m + n:
                if j <= ii < j + args.div:   #train1
                    f_w.write(label + ',' + text + '\n')
            elif m + n - j < args.div:
                if j <= ii < m + n or m <= ii < j + args.div - n:
                    f_w.write(label + ',' + text + '\n')
    if k+args.div <= m:
        print("train{}:{}-{}".format(i, k, k + args.div))
    elif m - k < args.div:
        print("train{}:{}-{},{}-{}".format(i, k, m, 0, args.div - m + k))

    if j + args.div <= m + n:
        print("train{}:{}-{}".format(i, j, j + args.div))
    elif m + n - j < args.div:
        print("train{}:{}-{},{}-{}".format(i, j, m + n, m, j + args.div - n))

    k += args.div
    if k > m:
        k = k - m
    j += args.div
    if j > m + n:
        j = j - n

print('2div_train down')
# 800 1036
# ----------
# train1:0-100
# train1:801-1103
# ----------
# train2:100-200
# train2:1103-1203
# ----------
# train3:200-300
# train3:1203-1303
# ----------
# train4:300-400
# train4:1303-1403
# ----------
# train5:400-500
# train5:1403-1503
# ----------
# train6:500-600
# train6:1503-1603
# ----------
# train7:600-700
# train7:1603-1703
# ----------
# train8:700-800
# train8:1703-1803
# ----------
# train9:0-100
# train9:1803-1903
# ----------
# train10:100-200
# train10:1903-2003




