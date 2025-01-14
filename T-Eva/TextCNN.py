import logging
from sklearn.metrics import precision_score, recall_score, f1_score

import torch.optim as optim
from torch.utils.data import DataLoader

from T_Eva.cnn.model import TextCNN
from T_Eva.utils.textcnn_data import build_dict, NewsDataSet, CATEGIRY_LIST
from T_Eva.utils.textcnn_trainer import train, evaluate, predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


class TextCnn:
    def __init__(self, train_file, valid_file, test_file, test_new_file, num_words=5000, batch_size=32, lr=0.003, epochs=5):
        self.train_file = train_file
        self.valid_file = valid_file
        self.test_file = test_file
        self.test_new_file = test_new_file
        self.num_words = num_words
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.model = None
        self.dct = None

    def load_data(self):
        logger.info('load and preprocess data...')

        # 构建字典
        self.dct = build_dict([self.train_file, self.valid_file], num_words=self.num_words)

        # 构建数据集和数据加载器
        self.train_ds = NewsDataSet(self.train_file, self.dct)
        self.train_dl = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

        self.valid_ds = NewsDataSet(self.valid_file, self.dct)
        self.valid_dl = DataLoader(self.valid_ds, batch_size=self.batch_size)

        self.test_ds = NewsDataSet(self.test_file, self.dct)
        self.test_dl = DataLoader(self.test_ds, batch_size=self.batch_size)

        self.test_new_ds = NewsDataSet(self.test_new_file, self.dct)
        self.test_new_dl = DataLoader(self.test_new_ds, batch_size=self.batch_size)

    def build_model(self):
        logger.info('building model...')
        self.model = TextCNN(class_num=len(CATEGIRY_LIST), embed_size=len(self.dct))

    def train_model(self):
        logger.info('training model...')
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        train(self.model, optimizer, self.train_dl, self.valid_dl, epochs=self.epochs)

    def evaluate_model(self):
        logger.info('evaluating model...')
        loss, acc = evaluate(self.model, self.valid_dl)
        logger.info(f'Validation - Loss: {loss:.4f}, Accuracy: {acc:.4f}')
        return loss, acc

    def predict_and_evaluate(self):
        logger.info('predicting and evaluating...')
        # 对 T_test 数据集
        y_pred = predict(self.model, self.test_dl)
        y_true = self.test_ds.labels

        test_precision = precision_score(y_true, y_pred, average='weighted')
        test_recall = recall_score(y_true, y_pred, average='weighted')
        test_f1 = f1_score(y_true, y_pred, average='weighted')
        test_acc = (y_true == y_pred).sum() / y_pred.shape[0]

        logger.info(f'Ori. - Accuracy: {test_acc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1-Score: {test_f1:.4f}')

        # 对 T_newtest 数据集
        y_new_pred = predict(self.model, self.test_new_dl)
        y_new_true = self.test_new_ds.labels

        test_new_precision = precision_score(y_new_true, y_new_pred, average='weighted')
        test_new_recall = recall_score(y_new_true, y_new_pred, average='weighted')
        test_new_f1 = f1_score(y_new_true, y_new_pred, average='weighted')
        test_new_acc = (y_new_true == y_new_pred).sum() / y_new_pred.shape[0]

        logger.info(f'T-Eva - Accuracy: {test_new_acc:.4f}, Precision: {test_new_precision:.4f}, Recall: {test_new_recall:.4f}, F1-Score: {test_new_f1:.4f}')

        return {
            'Ori.': {
                'accuracy': test_acc,
                'precision': test_precision,
                'recall': test_recall,
                'f1': test_f1
            },
            'T-Eva': {
                'accuracy': test_new_acc,
                'precision': test_new_precision,
                'recall': test_new_recall,
                'f1': test_new_f1
            }
        }
