import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import pandas as pd
import numpy as np
import jieba

class SCLPipeline:
    """
    一个用于处理文本分类任务的 Pipeline 类。
    包括数据加载、预处理、特征提取、网络训练和模型评估。
    """

    def __init__(self, source_file, target_file, validation_file):
        self.source_file = source_file
        self.target_file = target_file
        self.validation_file = validation_file

    def load_data(self):
        """加载数据"""
        source_df = pd.read_csv(self.source_file, header=0, names=['label', 'text'])
        target_df = pd.read_csv(self.target_file, header=0, names=['label', 'text'])
        validation_df = pd.read_csv(self.validation_file, header=0, names=['label', 'text'])

        source_data = source_df['text'].tolist()
        source_labels = source_df['label'].tolist()
        target_data = target_df['text'].tolist()
        target_labels = target_df['label'].tolist()
        validation_data = validation_df['text'].tolist()
        validation_labels = validation_df['label'].tolist()

        return source_data, source_labels, target_data, target_labels, validation_data, validation_labels

    # @staticmethod
    # def preprocess_text(data):
    #     """对文本进行分词"""
    #     return [" ".join(jieba.cut(str(text))) for text in data]

    @staticmethod
    def preprocess_text(data):
        """文本预处理"""
        return [" ".join(text.split()) for text in data]

    @staticmethod
    def extract_features(source_data, target_data, max_features=50000):
        """特征提取"""
        vectorizer = TfidfVectorizer(max_features=max_features)
        combined_data = source_data + target_data
        vectorizer.fit(combined_data)
        X_s = vectorizer.transform(source_data).toarray()
        X_t = vectorizer.transform(target_data).toarray()
        return X_s, X_t, vectorizer

    @staticmethod
    def select_pivot_features(X_s, y_s, top_k=100):
        """选择桥接特征"""
        pca = PCA(n_components=10)
        X_s_reduced = pca.fit_transform(X_s)
        mutual_info = mutual_info_classif(X_s_reduced, y_s, discrete_features=False)
        pivot_indices = np.argsort(mutual_info)[-top_k:]
        return pivot_indices

    @staticmethod
    def save_model(model, model_name):
        """保存模型"""
        model_path = os.path.join(os.getcwd(), f"{model_name}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Saved {model_name} to {model_path}")

    @staticmethod
    def train_auxiliary_net(X_s, X_pivot, input_dim, pivot_dim, epochs=10, batch_size=2, lr=0.001):
        """训练辅助任务网络"""
        class AuxiliaryNet(nn.Module):
            def __init__(self, input_dim, pivot_dim):
                super(AuxiliaryNet, self).__init__()
                self.fc = nn.Linear(input_dim, pivot_dim)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                return self.sigmoid(self.fc(x))

        aux_net = AuxiliaryNet(input_dim, pivot_dim)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(aux_net.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1, verbose=True)

        dataset = TensorDataset(torch.tensor(X_s, dtype=torch.float32), torch.tensor(X_pivot, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_loss = float('inf')
        best_model = None

        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = aux_net(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch + 1}, Loss: {avg_loss}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model = aux_net.state_dict()

            scheduler.step(avg_loss)

        aux_net.load_state_dict(best_model)
        SCLPipeline.save_model(aux_net, "auxiliary_net")
        return aux_net

    @staticmethod
    def train_target_net(X_shared, y_s, shared_dim, hidden_dim, output_dim, epochs=200, batch_size=64, lr=0.001):
        """训练目标任务网络"""
        class TargetNet(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super(TargetNet, self).__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.bn1 = nn.BatchNorm1d(hidden_dim)
                self.relu1 = nn.ReLU()
                self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
                self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
                self.relu2 = nn.ReLU()
                self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
                self.dropout = nn.Dropout(0.1)

            def forward(self, x):
                x = self.relu1(self.bn1(self.fc1(x)))
                x = self.dropout(x)
                x = self.relu2(self.bn2(self.fc2(x)))
                x = self.dropout(x)
                return self.fc3(x)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        target_net = TargetNet(shared_dim, hidden_dim, output_dim)
        target_net.apply(init_weights)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(target_net.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1, verbose=True)

        dataset = TensorDataset(torch.tensor(X_shared, dtype=torch.float32), torch.tensor(y_s, dtype=torch.float32).view(-1, 1))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_loss = float('inf')
        best_model = None

        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = target_net(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch + 1}, Loss: {avg_loss}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model = target_net.state_dict()

            scheduler.step(avg_loss)

        target_net.load_state_dict(best_model)
        SCLPipeline.save_model(target_net, "target_net")
        return target_net

    def run(self, aux_params=None, target_params=None):
        """运行主流程，支持修改辅助模型和目标模型的超参数"""
        if aux_params is None:
            aux_params = {'epochs': 10, 'batch_size': 2, 'lr': 0.001}
        if target_params is None:
            target_params = {'epochs': 200, 'batch_size': 64, 'lr': 0.001, 'hidden_dim': 500, 'output_dim': 1}

        source_data, source_labels, target_data, target_labels, validation_data, validation_labels = self.load_data()

        source_data = SCLPipeline.preprocess_text(source_data)
        target_data = SCLPipeline.preprocess_text(target_data)
        validation_data = SCLPipeline.preprocess_text(validation_data)

        X_s, X_t, vectorizer = SCLPipeline.extract_features(source_data, target_data)
        y_s = np.array(source_labels, dtype=np.float32)

        pivot_indices = SCLPipeline.select_pivot_features(X_s, y_s, top_k=1000)
        X_pivot = X_s[:, pivot_indices]

        aux_net = SCLPipeline.train_auxiliary_net(X_s, X_pivot, input_dim=X_s.shape[1], pivot_dim=X_pivot.shape[1], **aux_params)

        with torch.no_grad():
            shared_features = aux_net.fc.weight.data.numpy()
        X_shared = (X_s @ shared_features.T).astype(np.float32)
        X_t_shared = (X_t @ shared_features.T).astype(np.float32)
        X_v_shared = vectorizer.transform(validation_data).toarray() @ shared_features.T

        target_net = SCLPipeline.train_target_net(X_shared, y_s, shared_dim=shared_features.shape[0], **target_params)

        X_t_shared_tensor = torch.tensor(X_t_shared, dtype=torch.float32)
        with torch.no_grad():
            predictions = target_net(X_t_shared_tensor)

        binary_predictions = (predictions.numpy() >= 0.5).astype(int)
        print("Binary Predictions on target domain:", binary_predictions)

        acc = accuracy_score(target_labels, binary_predictions)
        f1 = f1_score(target_labels, binary_predictions)
        rec = recall_score(target_labels, binary_predictions)
        pre = precision_score(target_labels, binary_predictions)
        print(f"Accuracy on target domain: {acc:.4f}")
        print(f"F1 Score on target domain: {f1:.4f}")
        print(f"Recall on target domain: {rec:.4f}")
        print(f"Precision on target domain: {pre:.4f}")

        X_v_shared_tensor = torch.tensor(X_v_shared, dtype=torch.float32)
        with torch.no_grad():
            validation_predictions = target_net(X_v_shared_tensor)

        binary_validation_predictions = (validation_predictions.numpy() >= 0.5).astype(int)
        print("Binary Predictions on validation domain:", binary_validation_predictions)

        validation_acc = accuracy_score(validation_labels, binary_validation_predictions)
        validation_f1 = f1_score(validation_labels, binary_validation_predictions)
        validation_rec = recall_score(validation_labels, binary_validation_predictions)
        validation_pre = precision_score(validation_labels, binary_validation_predictions)
        print(f"Accuracy on validation domain: {validation_acc:.4f}")
        print(f"F1 Score on validation domain: {validation_f1:.4f}")
        print(f"Recall on validation domain: {validation_rec:.4f}")
        print(f"Precision on validation domain: {validation_pre:.4f}")

        with open("metrics.txt", "a") as f:
            f.write("Target Domain Metrics:\n")
            f.write(f"Accuracy: {acc:.4f}\t")
            f.write(f"F1 Score: {f1:.4f}\t")
            f.write(f"Recall: {rec:.4f}\t")
            f.write(f"Precision: {pre:.4f}\n")

            f.write("Validation Domain Metrics:\n")
            f.write(f"Accuracy: {validation_acc:.4f}\t")
            f.write(f"F1 Score: {validation_f1:.4f}\t")
            f.write(f"Recall: {validation_rec:.4f}\t")
            f.write(f"Precision: {validation_pre:.4f}\n\n")
