import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from T_Eva.utils.utils import flatten, get_batch
from T_Eva.utils.autoencoder import AutoEncoder


class SDA:
    def __init__(self, learning_rate=0.001, batch_size=32, latent_dim=40, epochs=5, train_size=1000):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.train_size = train_size
        self.autoencoder = None
        self.logistic_model = None
        self.vectorizer = CountVectorizer(min_df=1)  # 默认 CUT_OFF=1

    def load_data(self, train_path, test_path):
        # 加载数据
        def read_file(path):
            label, y, text, length = [], [], [], []
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.split(',')
                    label.append(line[0])
                    y.append(line[1])
                    text.append(line[2])
                    length.append(line[3].replace('\n', ''))
            return pd.DataFrame({'label': label, 'y': y, 'text': text, 'length': length})

        self.data = read_file(train_path)
        self.data_new = read_file(test_path)

    def preprocess_data(self):
        # 预处理 TF 矩阵
        tf = self.vectorizer.fit_transform(self.data['text'])
        tf_matrix = tf.toarray() * 1.0
        self.data['TF'] = [x.reshape(1, tf.shape[1]) for x in tf_matrix]

    def train_autoencoder_model(self, X, nb_epochs, lr, batch_size, original_dim, latent_dim):
        """
            Training an autoencoder with data X
            This method will return an autoencoder
        """
        autoencoder = AutoEncoder(original_dim, latent_dim)
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(nb_epochs):
            sum_loss = 0
            for step, x in enumerate(get_batch(X, batch_size)):
                vab_x = Variable(x)
                encoded, decoded = autoencoder(vab_x)
                loss = criterion(decoded, vab_x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                sum_loss += loss.item()
            print(f"Epoch {epoch + 1}: Loss = {sum_loss:.4f}")

        return autoencoder

    def train_autoencoder(self):
        # 训练 AutoEncoder
        X = self.data['TF']
        original_dim = X[0].shape[1]
        self.autoencoder = AutoEncoder(original_dim, self.latent_dim)
        self.autoencoder = self.train_autoencoder_model(X=X, nb_epochs=self.epochs, lr=self.learning_rate,
                                 batch_size=self.batch_size, original_dim=original_dim,
                                 latent_dim=self.latent_dim)

    def train_logistic_regression(self, train_size=1000):
        """
        使用编码后的数据训练 Logistic Regression，返回性能指标
        :param train_size: 训练集大小
        :return: accuracy, precision, recall, f1_score
        """
        # 使用编码后的数据
        X = self.data['TF']
        encoded_X, _ = self.autoencoder(Variable(torch.Tensor(flatten(X))))
        self.logistic_model = LogisticRegression(multi_class="ovr", solver='lbfgs')

        # 数据划分
        X_train, X_test, y_train, y_test = train_test_split(
            encoded_X.data.numpy(), self.data['y'], train_size=train_size, shuffle=False)

        # 训练模型
        self.logistic_model.fit(X_train, y_train)

        # 预测测试集
        y_pred = self.logistic_model.predict(X_test)

        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1_score = 2 * (precision * recall) / (precision + recall)

        # 输出指标
        print(f"Ori. - Accuracy: {accuracy:.5f}, Precision: {precision:.5f}, Recall: {recall:.5f}, F1-Score: {f1_score:.5f}")

        return accuracy, precision, recall, f1_score

    def predict_new_data(self):
        """
        使用训练好的模型对新数据进行预测并返回性能指标
        :return: accuracy, precision, recall, f1_score
        """
        # 检查是否已经训练了模型
        if self.autoencoder is None or self.logistic_model is None:
            raise ValueError("Autoencoder or Logistic Regression model is not trained. Please train the model first.")

        # 对新数据进行 TF 转换
        new_tf = self.vectorizer.transform(self.data_new['text'])
        new_tf_matrix = new_tf.toarray() * 1.0

        # 编码新数据
        encoded_X, _ = self.autoencoder(Variable(torch.Tensor(new_tf_matrix)))

        # 使用 Logistic Regression 模型进行预测
        y_pred = self.logistic_model.predict(encoded_X.data.numpy())

        # 获取真实标签
        y_true = self.data_new['y']

        # 计算性能指标
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1_score = 2 * (precision * recall) / (precision + recall)

        # 输出性能指标
        print(
            f"T-Eva - Accuracy: {accuracy:.5f}, Precision: {precision:.5f}, Recall: {recall:.5f}, F1-Score: {f1_score:.5f}")

        # 返回指标
        return accuracy, precision, recall, f1_score

    def run_pipeline(self, train_path, test_path):
        """
        一键运行完整 SDA 流程
        """
        print("------ 加载数据 ------")
        self.load_data(train_path, test_path)

        print("------ 数据预处理 ------")
        self.preprocess_data()

        print("------ 训练 AutoEncoder ------")
        self.train_autoencoder()

        print("------ 训练 Logistic Regression ------")
        self.train_logistic_regression(train_size=self.train_size)

        print("------ 新数据预测 ------")
        accuracy, precision, recall, f1_score = self.predict_new_data()
        return accuracy, precision, recall, f1_score
