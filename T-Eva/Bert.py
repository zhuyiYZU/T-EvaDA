import time
import torch
import numpy as np
from importlib import import_module
from T_Eva.utils.bert_utils import build_dataset, build_iterator, get_time_dif
from T_Eva.utils.bert_train_eval import train, new_eval


class BertPipeline:
    def __init__(self, dataset_path, model_name, random_seed=None):
        """
        初始化文本分类管道
        :param dataset_path: 数据集路径
        :param model_name: 模型名称 (Bert, ERNIE)
        :param random_seed: 随机种子 (可选)
        """
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.config = None
        self.model = None

        if random_seed is not None:
            self.set_random_seed(random_seed)

    def set_random_seed(self, seed):
        """
        设置随机种子以确保结果可重复
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    def load_model_and_config(self):
        """
        加载模型和配置
        """
        module = import_module('T_Eva.utils.' + self.model_name)
        self.config = module.Config(self.dataset_path)
        self.model = module.Model(self.config).to(self.config.device)

    def load_data(self):
        """
        加载数据并构建迭代器
        """
        print("Loading data...")
        self.train_data, self.dev_data, self.test_data, self.newtest_data = build_dataset(self.config)
        self.train_iter = build_iterator(self.train_data, self.config)
        self.dev_iter = build_iterator(self.dev_data, self.config)
        self.test_iter = build_iterator(self.test_data, self.config)
        self.newtest_iter = build_iterator(self.newtest_data, self.config)

        time_dif = get_time_dif(time.time())
        print("Time usage:", time_dif)

    def train_model(self):
        """
        训练模型
        """
        print("Training model...")
        train(self.config, self.model, self.train_iter, self.dev_iter, self.test_iter)

    def evaluate_new_test(self):
        """
        在新测试集上进行评估
        """
        print("Evaluating on new test data...")
        new_eval(self.config, self.model, self.newtest_iter)
