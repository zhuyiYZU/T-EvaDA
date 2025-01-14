from T_Eva.DSR import DSR

source_FILE = "datasets/amazon_d-e/train_all.csv"
target_FILE = "datasets/amazon_d-e/test_all.csv"
newtest_FILE = "datasets/amazon_d-e/newtest_all.csv"
INPUT_DIM = 2048  # 输入特征大小
learning_rate = 0.001
batch_size = 128
num_steps = 10000  # 训练轮次
NUM_CLASS = 2  # 类个数

dsr = DSR(input_feature_size=INPUT_DIM,
          source_file=source_FILE,target_file=target_FILE,newfile=newtest_FILE,
          learning_rate=learning_rate,batch_size=batch_size,
          num_steps=num_steps,NUM_CLASS=NUM_CLASS)  # 可以调整学习率
dsr.run()