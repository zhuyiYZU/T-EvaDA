import tqdm
import torch
import argparse
import time
import numpy as np
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, PtuningTemplate
from openprompt.prompts import ManualVerbalizer, KnowledgeableVerbalizer, SoftVerbalizer, AutomaticVerbalizer
from openprompt import PromptDataLoader, PromptForClassification
from openprompt.utils.reproduciblity import set_seed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sympy.sets.sets import set_function



class Spot:
    def __init__(self, model_name, model_path, dataset, template_id, verbalizer_type, seed=144, max_epochs=5, batch_size=32):
        """
        初始化 PromptModelPipeline
        :param model_name: 预训练模型名称（如 'bert'）
        :param model_path: 预训练模型路径
        :param dataset: 数据集名称
        :param template_id: 模板 ID
        :param verbalizer_type: verbalizer 类型（如 'manual', 'kpt', 'soft' 等）
        :param seed: 随机种子
        :param max_epochs: 最大训练轮次
        :param batch_size: 批次大小
        """
        self.model_name = model_name
        self.model_path = model_path
        self.dataset = dataset
        self.template_id = template_id
        self.verbalizer_type = verbalizer_type
        self.seed = seed
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.plm = None
        self.tokenizer = None
        self.prompt_model = None
        self.train_dataloader = None
        self.validation_dataloader = None
        self.test_dataloader = None
        set_seed(self.seed)
        dataset_name = self.dataset


    def load_model_and_data(self):
        """
        加载预训练语言模型和数据集
        """
        print("Loading model and dataset...")
        self.plm, self.tokenizer, model_config, WrapperClass = load_plm(self.model_name, self.model_path)
        global dataset_name
        # 根据数据集名称加载数据
        if self.dataset == "hotel-waimai":
            dataset_name = self.dataset
            from openprompt.data_utils.text_classification_dataset import Chinese_reviewProcessor
            processor = Chinese_reviewProcessor()
            dataset_path = "../datasets/Spot/hotel-waimai/"
            self.dataset = {
                'train': processor.get_train_examples(dataset_path),
                'test': processor.get_test_examples(dataset_path),
                'newtest':processor.get_newtest_examples(dataset_path)
            }
            self.class_labels = processor.get_labels()
        else:
            raise NotImplementedError("The dataset is not implemented.")

        # 加载模板
        self.template = PtuningTemplate(model=self.plm, tokenizer=self.tokenizer).from_file(
            "/home/ubuntu/Lizhenglong/T-Eva/T_Eva/scripts/TextClassification/hotel-waimai/ptuning_template.txt", choice=self.template_id
        )

        # 加载 Verbalizer
        if self.verbalizer_type == "manual":
            self.verbalizer = ManualVerbalizer(self.tokenizer, classes=self.class_labels).from_file(
                f"/home/ubuntu/Lizhenglong/T-Eva/T_Eva/scripts/TextClassification/hotel-waimai/manual_verbalizer.txt"
            )
        elif self.verbalizer_type == "kpt":
            self.verbalizer = KnowledgeableVerbalizer(self.tokenizer, classes=self.class_labels).from_file(
                f"./scripts/TextClassification/{self.dataset}/knowledgeable_verbalizer.txt"
            )
        elif self.verbalizer_type == "soft":
            self.verbalizer = SoftVerbalizer(self.tokenizer, plm=self.plm, classes=self.class_labels).from_file(
                f"./scripts/TextClassification/{self.dataset}/manual_verbalizer.txt"
            )
        else:
            raise ValueError("Invalid verbalizer type!")

        # 初始化 Prompt 模型
        self.prompt_model = PromptForClassification(plm=self.plm, template=self.template, verbalizer=self.verbalizer)
        self.prompt_model = self.prompt_model.cuda() if torch.cuda.is_available() else self.prompt_model
        from openprompt.data_utils.data_sampler import FewShotSampler
        sampler = FewShotSampler(num_examples_per_label=20, also_sample_dev=True,
                                 num_examples_per_label_dev=20)
        self.dataset['train'], self.dataset['validation'] = sampler(self.dataset['train'], seed=self.seed)
        # 加载数据
        self.train_dataloader = PromptDataLoader(
            dataset=self.dataset["train"], template=self.template, tokenizer=self.tokenizer,
            tokenizer_wrapper_class=WrapperClass, max_seq_length=128, batch_size=self.batch_size, shuffle=True
        )
        self.validation_dataloader = PromptDataLoader(
            dataset=self.dataset["validation"], template=self.template, tokenizer=self.tokenizer,
            tokenizer_wrapper_class=WrapperClass, max_seq_length=128, batch_size=self.batch_size, shuffle=False
        )
        self.test_dataloader = PromptDataLoader(
            dataset=self.dataset["test"], template=self.template, tokenizer=self.tokenizer,
            tokenizer_wrapper_class=WrapperClass, max_seq_length=128, batch_size=self.batch_size, shuffle=False
        )

        self.newtest_dataloader = PromptDataLoader(
            dataset=self.dataset["newtest"], template=self.template, tokenizer=self.tokenizer,
            tokenizer_wrapper_class=WrapperClass, max_seq_length=128, batch_size=self.batch_size, shuffle=False
        )

    def train(self):
        """
        训练 Prompt 模型
        """
        print("Training...")
        optimizer = torch.optim.AdamW(self.prompt_model.parameters(), lr=3e-5)
        loss_func = torch.nn.CrossEntropyLoss()
        best_val_acc = 0

        for epoch in range(self.max_epochs):
            self.prompt_model.train()
            total_loss = 0
            for step, inputs in enumerate(self.train_dataloader):
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                logits = self.prompt_model(inputs)
                labels = inputs['label']
                loss = loss_func(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.prompt_model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()

            # 验证集评估
            val_acc = self.evaluate1(self.validation_dataloader, desc="Validation")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.prompt_model.state_dict(), f"/home/ubuntu/Lizhenglong/T-Eva/T_Eva/ckpts/{dataset_name}.ckpt")

            print(f"Epoch {epoch + 1}: Loss = {total_loss:.4f}, Validation Accuracy = {val_acc:.4f}")

    def evaluate1(self, dataloader, desc="Evaluation"):
        """
        模型评估
        """
        self.prompt_model.eval()
        all_preds = []
        all_labels = []

        for inputs in tqdm.tqdm(dataloader, desc=desc):
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            logits = self.prompt_model(inputs)
            labels = inputs['label']
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

        acc = accuracy_score(all_labels, all_preds)
        return acc

    def evaluate2(self, dataloader, desc="Evaluation"):
        """
        模型评估
        """
        self.prompt_model.eval()
        all_preds = []
        all_labels = []

        for inputs in tqdm.tqdm(dataloader, desc=desc):
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            logits = self.prompt_model(inputs)
            labels = inputs['label']
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

        acc = accuracy_score(all_labels, all_preds)
        pre = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        print(f"Ori.  -Accuracy: {acc:.4f}, Precision: {pre:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
        return acc

    def evaluate3(self, dataloader, desc="Evaluation"):
        """
        模型评估
        """
        self.prompt_model.eval()
        all_preds = []
        all_labels = []

        for inputs in tqdm.tqdm(dataloader, desc=desc):
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            logits = self.prompt_model(inputs)
            labels = inputs['label']
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

        acc = accuracy_score(all_labels, all_preds)
        pre = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        print(f"T-Eva  -Accuracy: {acc:.4f}, Precision: {pre:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
        return acc

    def test(self):
        """
        测试 Prompt 模型
        """
        print("Testing...")
        self.prompt_model.load_state_dict(torch.load(f"/home/ubuntu/Lizhenglong/T-Eva/T_Eva/ckpts/{dataset_name}.ckpt"))
        self.evaluate2(self.test_dataloader, desc="Test")

    def test_new(self):
        """
        测试 Prompt 模型
        """
        print("Testing T-Eva...")
        self.prompt_model.load_state_dict(torch.load(f"/home/ubuntu/Lizhenglong/T-Eva/T_Eva/ckpts/{dataset_name}.ckpt"))
        self.evaluate3(self.newtest_dataloader, desc="Test")


