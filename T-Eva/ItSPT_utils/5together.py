import time
import argparse
parser = argparse.ArgumentParser("")
parser.add_argument("--dataset", type=str)
parser.add_argument("--itera", type=int)
parser.add_argument("--div", type=int)
args = parser.parse_args()

start_time = time.time()

itera = args.itera
votel1 = 7
votel2 = 6
num1 = 800
num2 = 1100
num3 = 1418

file = f"{args.dataset}_label1.csv"
count = 1
for count, line in enumerate(open(file, "r", encoding="utf-8").readlines()):
    count += 1

label_list = [0] * count
save_path = "result.csv"
for index in range(1, int(itera) + 1):
    read_path = f"{args.dataset}_label"+str(index)+".csv"
    with open(read_path,"r",encoding="utf-8") as fb:
        lines = fb.readlines()
        index_temp = 0
        for line in lines:
            label = line.split("\n")[0]
            label_list[index_temp] += int(label)
            index_temp += 1
    fb.close()
for i in range(count):
    with open(save_path,"a+",encoding="utf-8") as fb:
        fb.write(str(label_list[i])+"\n")
    fb.close()

t1, t2, t3, t4 = 0, 0, 0, 0
f1, f2, f3, f4 = 0, 0, 0, 0
read_path = "result.csv"
save_path = "result1.csv"

with open(read_path,"r",encoding="utf-8") as fb:
    # s = 0
    for ii, label in enumerate(fb):
        if 0 <= ii < num1:
            if label.split("\n")[0] == str(-votel1) :
                t1 += 1
                with open(save_path, "a+", encoding="utf-8") as fb:
                    fb.write(str(ii+1)+"\n")

with open(read_path,"r",encoding="utf-8") as fb:
    for ii, label in enumerate(fb):
        if 0 <= ii < num1:
            if label.split("\n")[0] == str(votel1) :
                f1 += 1
                with open(save_path, "a+", encoding="utf-8") as fb:
                    fb.write(str(ii+1)+"\n")

with open(read_path,"r",encoding="utf-8") as fb:
    for ii, label in enumerate(fb):
        if num1 <= ii < num2:
            if label.split("\n")[0] == str(-(votel1 + 1)) :
                t2 += 1
                with open(save_path, "a+", encoding="utf-8") as fb:
                    fb.write(str(ii+1)+"\n")

with open(read_path,"r",encoding="utf-8") as fb:
    for ii, label in enumerate(fb):
        if num1 <= ii < num2:
            if label.split("\n")[0] == str(votel1 + 1) :
                f2 += 1
                with open(save_path, "a+", encoding="utf-8") as fb:
                    fb.write(str(ii+1)+"\n")

with open(read_path, "r", encoding="utf-8") as fb:
    for ii, label in enumerate(fb):
        if num2 <= ii < num3:
            if label.split("\n")[0] == str(votel2):
                t3 += 1
                with open(save_path, "a+", encoding="utf-8") as fb:
                    fb.write(str(ii+1)+"\n")

with open(read_path, "r", encoding="utf-8") as fb:
    for ii, label in enumerate(fb):
        if num2 <= ii < num3:
            if label.split("\n")[0] == str(-votel2) :
                f3 += 1
                with open(save_path, "a+", encoding="utf-8") as fb:
                    fb.write(str(ii+1)+"\n")

with open(read_path, "r", encoding="utf-8") as fb:
    for ii, label in enumerate(fb):
        if num3 <= ii < count:
            if label.split("\n")[0] == str(votel2 + 1):
                t4 += 1
                with open(save_path, "a+", encoding="utf-8") as fb:
                    fb.write(str(ii + 1)+"\n")

with open(read_path, "r", encoding="utf-8") as fb:
    for ii, label in enumerate(fb):
        if num3 <= ii < count:
            if label.split("\n")[0] == str(-(votel2 + 1)):
                f4 += 1
                with open(save_path, "a+", encoding="utf-8") as fb:
                    fb.write(str(ii + 1)+"\n")

print(t1, f1, t2, f2, t3, f3, t4, f4)
print('1:{}'.format(t1+t2))
print('2:{}'.format(t3+t4))
T = t1+t2+t3+t4
print("True:{}".format(T))

print('2:{}'.format(f1+f2))
print('1:{}'.format(f3+f4))
F = f1+f2+f3+f4
print("False:{}".format(F))
print("Total:{}".format(T+F))

read_path1 = "result1.csv"
read_path2 = f"{args.dataset}_test.csv"
save_path = "train_all.csv"

hang_index = []
with open(read_path1, 'r', encoding='utf-8') as fa:
    for i, line in enumerate(fa):
        line = line.split('\n')[0]
        hang_index.append(int(line))
# print(hang_index)
with open(read_path2, 'r', encoding='utf-8') as fb:
    for i, line in enumerate(fb):

        if i+1 in hang_index[0: t1]:
            with open(save_path, 'a+', encoding='utf-8') as fc:
                fc.write('"1",' + line)
        if i+1 in hang_index[t1: t1+f1]:
            with open(save_path, 'a+', encoding='utf-8') as fc:
                fc.write('"2",' + line)
        if i+1 in hang_index[t1+f1: t1+f1+t2]:
            with open(save_path, 'a+', encoding='utf-8') as fc:
                fc.write('"1",' + line)
        if i+1 in hang_index[t1+f1+t2: t1+f1+t2+f2]:
            with open(save_path, 'a+', encoding='utf-8') as fc:
                fc.write('"2",' + line)
        if i+1 in hang_index[t1+f1+t2+f2: t1+f1+t2+f2+t3]:
            with open(save_path, 'a+', encoding='utf-8') as fc:
                fc.write('"2",' + line)
        if i+1 in hang_index[t1+f1+t2+f2+t3: t1+f1+t2+f2+t3+f3]:
            with open(save_path, 'a+', encoding='utf-8') as fc:
                fc.write('"1",' + line)
        if i+1 in hang_index[t1+f1+t2+f2+t3+f3: t1+f1+t2+f2+t3+f3+t4]:
            with open(save_path, 'a+', encoding='utf-8') as fc:
                fc.write('"2",' + line)
        if i+1 in hang_index[t1+f1+t2+f2+t3+f3+t4: ]:
            with open(save_path, 'a+', encoding='utf-8') as fc:
                fc.write('"1",' + line)

end_time = time.time()
run_time = end_time - start_time
with open('time.csv', 'a+', encoding='utf-8') as t:
    t.write(f'{args.dataset}, step5 runtime: ' + str(run_time) + '\n')







