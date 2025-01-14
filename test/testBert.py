from T_Eva.Bert import BertPipeline

if __name__ == "__main__":
    dataset_path = 'datasets/hotel-weibo'
    model_name = 'bert'  # 或者 'ERNIE'
    # 初始化管道
    pipeline = BertPipeline(dataset_path, model_name, random_seed=42)
    # 加载模型和配置
    pipeline.load_model_and_config()
    # 加载数据
    pipeline.load_data()
    # 训练模型
    pipeline.train_model()
    # 在新测试集上评估
    pipeline.evaluate_new_test()
