from T_Eva.SDA import SDA

sda = SDA(learning_rate=0.001, train_size=800)  # 可以调整学习率
predictions = sda.run_pipeline("datasets/hotel-waimai/A_train.csv", "datasets/hotel-waimai/A_newtest.csv")

