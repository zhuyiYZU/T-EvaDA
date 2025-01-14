from T_Eva.TextCNN import TextCnn
if __name__ == "__main__":
    # 数据文件路径
    train_file = "datasets/weibo-waimai/T_train.csv"
    valid_file = "datasets/weibo-waimai/T_val.csv"
    test_file = "datasets/weibo-waimai/T_test.csv"
    test_new_file = "datasets/weibo-waimai/T_newtest.csv"

    # 初始化Pipeline
    pipeline = TextCnn(train_file, valid_file, test_file, test_new_file, num_words=5000, batch_size=32, lr=0.001, epochs=5)
    # 加载数据
    pipeline.load_data()
    # 构建模型
    pipeline.build_model()
    # 训练模型
    pipeline.train_model()
    # 验证模型
    pipeline.evaluate_model()
    # 测试和评估
    metrics = pipeline.predict_and_evaluate()