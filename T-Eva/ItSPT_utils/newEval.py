import tqdm
# from openprompt.data_utils.text_classification_dataset import AgnewsProcessor, DBpediaProcessor, ImdbProcessor, AmazonProcessor, News_1Processor, K_EProcessor, AclProcessor
from openprompt.data_utils.text_classification_dataset import *
import torch
from openprompt.data_utils.utils import InputExample
import argparse
import numpy as np
import time
from sklearn.metrics import *
from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer, KnowledgeableVerbalizer, SoftVerbalizer, AutomaticVerbalizer
from openprompt.prompts import ManualTemplate, PtuningTemplate
from openprompt.utils import metrics

start_time = time.time()
# 创建一个解析对象；
parser = argparse.ArgumentParser("")
# 然后向该对象中添加你要关注的命令行参数和选项，每一个add_argument方法对应一个你要关注的参数或选项；
parser.add_argument("--shot", type=int, default=10)
parser.add_argument("--seed", type=int, default=144)
parser.add_argument("--plm_eval_mode", action="store_true")
parser.add_argument("--model", type=str, default='bert')
parser.add_argument("--model_name_or_path", default='/home/ubuntu/Lizhenglong/chinese_itera/model')
# parser.add_argument("--model_name_or_path", default='bert-base-chinese')
parser.add_argument("--verbalizer", type=str)
parser.add_argument("--calibration", action="store_true")
parser.add_argument("--filter", default="none", type=str)
parser.add_argument("--template_id", type=int)
parser.add_argument("--dataset", type=str)
parser.add_argument("--result_file", type=str, default="../sfs_scripts/results_fewshot_manual_kpt.txt")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--max_epochs", type=int, default=5)
parser.add_argument("--kptw_lr", default=0.06, type=float)
parser.add_argument("--pred_temp", default=1.0, type=float)
parser.add_argument("--max_token_split", default=-1, type=int)
parser.add_argument("--batch_size", default=32, type=int)
# 最后调用parse_args()方法进行解析；解析成功之后即可使用。
args = parser.parse_args()

import random

this_run_unicode = str(random.randint(0, 1e10))

from openprompt.utils.reproduciblity import set_seed

set_seed(args.seed)

# 第二步：定义预训练语言模型作为主干
from openprompt.plms import load_plm

# 确定预训练语言模型
plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)  # 这里选择的预训练语言模型是bert

dataset = {}

if args.dataset == "agnews":
    dataset['train'] = AgnewsProcessor().get_train_examples("./datasets/TextClassification/agnews/")
    dataset['test'] = AgnewsProcessor().get_test_examples("./datasets/TextClassification/agnews/")
    class_labels = AgnewsProcessor().get_labels()
    scriptsbase = "TextClassification/agnews"
    scriptformat = "txt"
    cutoff = 0.5
    max_seq_l = 128
    batch_s = args.batch_size
elif args.dataset == "CT_Hownet":
    dataset['train'] = JD().get_train_examples("./datasets/TextClassification/CT_Hownet/")
    dataset['test'] = JD().get_test_examples("./datasets/TextClassification/CT_Hownet/")
    class_labels = JD().get_labels()
    scriptsbase = "TextClassification/CT_Hownet"
    scriptformat = "txt"
    cutoff = 0.5
    max_seq_l = 128
    batch_s = args.batch_size
elif args.dataset == "newsgroups1":
    dataset['train'] = News_1Processor().get_train_examples("./datasets/TextClassification/newsgroups1/")
    dataset['test'] = News_1Processor().get_test_examples("./datasets/TextClassification/newsgroups1/")
    class_labels =News_1Processor().get_labels()
    scriptsbase = "TextClassification/newsgroups1"
    scriptformat = "txt"
    cutoff=0.5
    max_seq_l = 128
    batch_s = 32
elif args.dataset == "newsgroups2":
    dataset['train'] = News_2Processor().get_train_examples("./datasets/TextClassification/newsgroups2/")
    dataset['test'] = News_2Processor().get_test_examples("./datasets/TextClassification/newsgroups2/")
    class_labels =News_2Processor().get_labels()
    scriptsbase = "TextClassification/newsgroups2"
    scriptformat = "txt"
    cutoff=0.5
    max_seq_l = 128
    batch_s = 32
elif args.dataset == "newsgroups3":
    dataset['train'] = News_3Processor().get_train_examples("./datasets/TextClassification/newsgroups3/")
    dataset['test'] = News_3Processor().get_test_examples("./datasets/TextClassification/newsgroups3/")
    class_labels =News_3Processor().get_labels()
    scriptsbase = "TextClassification/newsgroups3"
    scriptformat = "txt"
    cutoff=0.5
    max_seq_l = 128
    batch_s = 32
elif args.dataset == "newsgroups4":
    dataset['train'] = News_1Processor().get_train_examples("./datasets/TextClassification/newsgroups4/")
    dataset['test'] = News_1Processor().get_test_examples("./datasets/TextClassification/newsgroups4/")
    class_labels =News_1Processor().get_labels()
    scriptsbase = "TextClassification/newsgroups4"
    scriptformat = "txt"
    cutoff=0.5
    max_seq_l = 128
    batch_s = 32
elif args.dataset == "newsgroups5":
    dataset['train'] = News_2Processor().get_train_examples("./datasets/TextClassification/newsgroups5/")
    dataset['test'] = News_2Processor().get_test_examples("./datasets/TextClassification/newsgroups5/")
    class_labels =News_2Processor().get_labels()
    scriptsbase = "TextClassification/newsgroups5"
    scriptformat = "txt"
    cutoff=0.5
    max_seq_l = 128
    batch_s = 32
elif args.dataset == "newsgroups6":
    dataset['train'] = News_3Processor().get_train_examples("./datasets/TextClassification/newsgroups6/")
    dataset['test'] = News_3Processor().get_test_examples("./datasets/TextClassification/newsgroups6/")
    class_labels =News_3Processor().get_labels()
    scriptsbase = "TextClassification/newsgroups6"
    scriptformat = "txt"
    cutoff=0.5
    max_seq_l = 128
    batch_s = 32
elif args.dataset == "JD_Morphonym":
    dataset['train'] = JD().get_train_examples("./datasets/TextClassification/JD_Morphonym/")
    dataset['test'] = JD().get_test_examples("./datasets/TextClassification/JD_Morphonym/")
    class_labels = JD().get_labels()
    scriptsbase = "TextClassification/JD_Morphonym"
    scriptformat = "txt"
    cutoff = 0.5
    max_seq_l = 128
    batch_s = args.batch_size
elif args.dataset == "amazon_e-k":
    dataset['train'] = K_EProcessor().get_train_examples("./datasets/TextClassification/amazon_e-k/")
    dataset['test'] = K_EProcessor().get_test_examples("./datasets/TextClassification/amazon_e-k/")
    class_labels =K_EProcessor().get_labels()
    scriptsbase = "TextClassification/amazon_e-k"
    scriptformat = "txt"
    cutoff=0.5
    max_seq_l = 128
    batch_s = 32
elif args.dataset == "JD_Hownet":
    dataset['train'] = JD().get_train_examples("./datasets/TextClassification/JD_Hownet/")
    dataset['test'] = JD().get_test_examples("./datasets/TextClassification/JD_Hownet/")
    class_labels = JD().get_labels()
    scriptsbase = "TextClassification/JD_Hownet"
    scriptformat = "txt"
    cutoff = 0.5
    max_seq_l = 128
    batch_s = args.batch_size
elif args.dataset == "JD_Comp":
    dataset['train'] = JD().get_train_examples("./datasets/TextClassification/JD_Comp/")
    dataset['test'] = JD().get_test_examples("./datasets/TextClassification/JD_Comp/")
    class_labels = JD().get_labels()
    scriptsbase = "TextClassification/JD_Comp"
    scriptformat = "txt"
    cutoff = 0.5
    max_seq_l = 128
    batch_s = args.batch_size
elif args.dataset == "JD_MLM":
    dataset['train'] = JD().get_train_examples("./datasets/TextClassification/JD_MLM/")
    dataset['test'] = JD().get_test_examples("./datasets/TextClassification/JD_MLM/")
    class_labels = JD().get_labels()
    scriptsbase = "TextClassification/JD_MLM"
    scriptformat = "txt"
    cutoff = 0.5
    max_seq_l = 128
    batch_s = args.batch_size
elif args.dataset == "CT_Morphonym":
    dataset['train'] = CT().get_train_examples("./datasets/TextClassification/CT_Morphonym/")
    dataset['test'] = CT().get_test_examples("./datasets/TextClassification/CT_Morphonym/")
    class_labels = CT().get_labels()
    scriptsbase = "TextClassification/CT_Morphonym"
    scriptformat = "txt"
    cutoff = 0.5
    max_seq_l = 128
    batch_s = args.batch_size
elif args.dataset == "CT_Comp":
    dataset['train'] = CT().get_train_examples("./datasets/TextClassification/CT_Comp/")
    dataset['test'] = CT().get_test_examples("./datasets/TextClassification/CT_Comp/")
    class_labels = CT().get_labels()
    scriptsbase = "TextClassification/CT_Comp"
    scriptformat = "txt"
    cutoff = 0.5
    max_seq_l = 128
    batch_s = args.batch_size
elif args.dataset == "CT_Homophone":
    dataset['train'] = CT().get_train_examples("./datasets/TextClassification/CT_Homophone/")
    dataset['test'] = CT().get_test_examples("./datasets/TextClassification/CT_Homophone/")
    class_labels = CT().get_labels()
    scriptsbase = "TextClassification/CT_Homophone"
    scriptformat = "txt"
    cutoff = 0.5
    max_seq_l = 128
    batch_s = args.batch_size
elif args.dataset == "CT_Hownet":
    dataset['train'] = JD().get_train_examples("./datasets/TextClassification/CT_Hownet/")
    dataset['test'] = JD().get_test_examples("./datasets/TextClassification/CT_Hownet/")
    class_labels = JD().get_labels()
    scriptsbase = "TextClassification/CT_Hownet"
    scriptformat = "txt"
    cutoff = 0.5
    max_seq_l = 128
    batch_s = args.batch_size
elif args.dataset == "CT_Comp":
    dataset['train'] = CT().get_train_examples("./datasets/TextClassification/CT_Comp/")
    dataset['test'] = CT().get_test_examples("./datasets/TextClassification/CT_Comp/")
    class_labels = CT().get_labels()
    scriptsbase = "TextClassification/CT_Comp"
    scriptformat = "txt"
    cutoff = 0.5
    max_seq_l = 128
    batch_s = args.batch_size

elif args.dataset == "hc3_finance":
    dataset['train'] = Human_gptProcessor().get_train_examples("datasets/TextClassification/hc3_finance/")
    dataset['test'] = Human_gptProcessor().get_test_examples("datasets/TextClassification/hc3_finance/")
    class_labels = Human_gptProcessor().get_labels()
    scriptsbase = "TextClassification/hc3_finance"
    scriptformat = "txt"
    cutoff = 0.5
    max_seq_l = 128
    batch_s = args.batch_size
elif args.dataset == "hotel-waimai":
    dataset['train'] = Chinese_reviewProcessor().get_train_examples("./datasets/TextClassification/hotel-waimai/")
    dataset['test'] = Chinese_reviewProcessor().get_test_examples("./datasets/TextClassification/hotel-waimai/")
    class_labels = Chinese_reviewProcessor().get_labels()
    scriptsbase = "TextClassification/hotel-waimai"
    scriptformat = "txt"
    cutoff = 0.5
    max_seq_l = 64
    batch_s = args.batch_size
elif args.dataset == "hotel-weibo":
    dataset['train'] = Chinese_reviewProcessor().get_train_examples("./datasets/TextClassification/hotel-weibo/")
    dataset['test'] = Chinese_reviewProcessor().get_test_examples("./datasets/TextClassification/hotel-weibo/")
    class_labels = Chinese_reviewProcessor().get_labels()
    scriptsbase = "TextClassification/hotel-weibo"
    scriptformat = "txt"
    cutoff = 0.5
    max_seq_l = 64
    batch_s = args.batch_size
elif args.dataset == "waimai-hotel":
    dataset['train'] = Chinese_reviewProcessor().get_train_examples("./datasets/TextClassification/waimai-hotel/")
    dataset['test'] = Chinese_reviewProcessor().get_test_examples("./datasets/TextClassification/waimai-hotel/")
    class_labels = Chinese_reviewProcessor().get_labels()
    scriptsbase = "TextClassification/waimai-hotel"
    scriptformat = "txt"
    cutoff = 0.5
    max_seq_l = 64
    batch_s = args.batch_size
elif args.dataset == "waimai-weibo":
    dataset['train'] = Chinese_reviewProcessor().get_train_examples("./datasets/TextClassification/waimai-weibo/")
    dataset['test'] = Chinese_reviewProcessor().get_test_examples("./datasets/TextClassification/waimai-weibo/")
    class_labels = Chinese_reviewProcessor().get_labels()
    scriptsbase = "TextClassification/waimai-weibo"
    scriptformat = "txt"
    cutoff = 0.5
    max_seq_l = 64
    batch_s = args.batch_size
elif args.dataset == "weibo-hotel":
    dataset['train'] = Chinese_reviewProcessor().get_train_examples("./datasets/TextClassification/weibo-hotel/")
    dataset['test'] = Chinese_reviewProcessor().get_test_examples("./datasets/TextClassification/weibo-hotel/")
    class_labels = Chinese_reviewProcessor().get_labels()
    scriptsbase = "TextClassification/weibo-hotel"
    scriptformat = "txt"
    cutoff = 0.5
    max_seq_l = 64
    batch_s = args.batch_size
elif args.dataset == "weibo-waimai":
    dataset['train'] = Chinese_reviewProcessor().get_train_examples("./datasets/TextClassification/weibo-waimai/")
    dataset['test'] = Chinese_reviewProcessor().get_test_examples("./datasets/TextClassification/weibo-waimai/")
    class_labels = Chinese_reviewProcessor().get_labels()
    scriptsbase = "TextClassification/weibo-waimai"
    scriptformat = "txt"
    cutoff = 0.5
    max_seq_l = 64
    batch_s = args.batch_size
elif args.dataset == "camera-car":
    dataset['train'] = Chinese_reviewProcessor().get_train_examples("./datasets/TextClassification/camera-car/")
    dataset['test'] = Chinese_reviewProcessor().get_test_examples("./datasets/TextClassification/camera-car/")
    class_labels = Chinese_reviewProcessor().get_labels()
    scriptsbase = "TextClassification/camera-car"
    scriptformat = "txt"
    cutoff = 0.5
    max_seq_l = 64
    batch_s = args.batch_size
elif args.dataset == "camera-notebook":
    dataset['train'] = Chinese_reviewProcessor().get_train_examples("./datasets/TextClassification/camera-notebook/")
    dataset['test'] = Chinese_reviewProcessor().get_test_examples("./datasets/TextClassification/camera-notebook/")
    class_labels = Chinese_reviewProcessor().get_labels()
    scriptsbase = "TextClassification/camera-notebook"
    scriptformat = "txt"
    cutoff = 0.5
    max_seq_l = 64
    batch_s = args.batch_size
elif args.dataset == "camera-phone":
    dataset['train'] = Chinese_reviewProcessor().get_train_examples("./datasets/TextClassification/camera-phone/")
    dataset['test'] = Chinese_reviewProcessor().get_test_examples("./datasets/TextClassification/camera-phone/")
    class_labels = Chinese_reviewProcessor().get_labels()
    scriptsbase = "TextClassification/camera-phone"
    scriptformat = "txt"
    cutoff = 0.5
    max_seq_l = 64
    batch_s = args.batch_size
elif args.dataset == "car-camera":
    dataset['train'] = Chinese_reviewProcessor().get_train_examples("./datasets/TextClassification/car-camera/")
    dataset['test'] = Chinese_reviewProcessor().get_test_examples("./datasets/TextClassification/car-camera/")
    class_labels = Chinese_reviewProcessor().get_labels()
    scriptsbase = "TextClassification/car-camera"
    scriptformat = "txt"
    cutoff = 0.5
    max_seq_l = 64
    batch_s = args.batch_size
elif args.dataset == "amazon_e-k":
    dataset['train'] = K_EProcessor().get_train_examples("./datasets/TextClassification/amazon_e-k/")
    dataset['test'] = K_EProcessor().get_test_examples("./datasets/TextClassification/amazon_e-k/")
    class_labels =K_EProcessor().get_labels()
    scriptsbase = "TextClassification/amazon_e-k"
    scriptformat = "txt"
    cutoff=0.5
    max_seq_l = 128
    batch_s = 32
elif args.dataset == "amazon_e-k":
    dataset['train'] = K_EProcessor().get_train_examples("./datasets/TextClassification/amazon_e-k/")
    dataset['test'] = K_EProcessor().get_test_examples("./datasets/TextClassification/amazon_e-k/")
    class_labels =K_EProcessor().get_labels()
    scriptsbase = "TextClassification/amazon_e-k"
    scriptformat = "txt"
    cutoff=0.5
    max_seq_l = 128
    batch_s = 32
elif args.dataset == "amazon_e-k":
    dataset['train'] = K_EProcessor().get_train_examples("./datasets/TextClassification/amazon_e-k/")
    dataset['test'] = K_EProcessor().get_test_examples("./datasets/TextClassification/amazon_e-k/")
    class_labels =K_EProcessor().get_labels()
    scriptsbase = "TextClassification/amazon_e-k"
    scriptformat = "txt"
    cutoff=0.5
    max_seq_l = 128
    batch_s = 32
else:
    raise NotImplementedError

# 第三步：模板选择
# mytemplate = ManualTemplate(tokenizer=tokenizer).from_file(f"./scripts/{scriptsbase}/manual_template.txt",
#                                                            choice=args.template_id)
mytemplate = PtuningTemplate(model=plm, tokenizer=tokenizer).from_file(f"./scripts/{scriptsbase}/ptuning_template.txt", choice=args.template_id)
# 第四步：定义Verbalizer是另一个重要的
#   Verbalizer将原始标签投射到一组label单词中，如把消极类投射到单词bad，把积极类投射到单词good，wonderful，great

if args.verbalizer == "kpt":
    myverbalizer = KnowledgeableVerbalizer(tokenizer, classes=class_labels, candidate_frac=cutoff,
                                           pred_temp=args.pred_temp, max_token_split=args.max_token_split).from_file(
        f"scripts/{scriptsbase}/knowledgeable_verbalizer.{scriptformat}")
elif args.verbalizer == "manual":
    myverbalizer = ManualVerbalizer(tokenizer, classes=class_labels).from_file(
        f"scripts/{scriptsbase}/manual_verbalizer.{scriptformat}")
elif args.verbalizer == "soft":
    myverbalizer = SoftVerbalizer(tokenizer, plm=plm, classes=class_labels).from_file(
        f"scripts/{scriptsbase}/manual_verbalizer.{scriptformat}")
elif args.verbalizer == "auto":
    myverbalizer = AutomaticVerbalizer(tokenizer, classes=class_labels)

# (contextual) calibration
if args.verbalizer in ["kpt", "manual"]:
    if args.calibration or args.filter != "none":
        from openprompt.data_utils.data_sampler import FewShotSampler

        support_sampler = FewShotSampler(num_examples_total=200, also_sample_dev=False)
        dataset['support'] = support_sampler(dataset['train'], seed=args.seed)

        # for example in dataset['support']:
        #     example.label = -1 # remove the labels of support set for clarification
        support_dataloader = PromptDataLoader(dataset=dataset["support"], template=mytemplate, tokenizer=tokenizer,
                                              tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                              decoder_max_length=3,
                                              batch_size=batch_s, shuffle=False, teacher_forcing=False,
                                              predict_eos_token=False,
                                              truncate_method="tail")

# 第五步：构造PromptModel，有三个对象，分别是：PLM,Prompt,Verbalizer
#    将它们合并到PromptModel中
#    给定任务，现在我们有一个PLM、一个模板和一个Verbalizer，我们将它们合并到PromptModel中

from openprompt import PromptForClassification

use_cuda = True
prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False,
                                       plm_eval_mode=args.plm_eval_mode)

if use_cuda:
    prompt_model = prompt_model.cuda()

# HP
# if args.calibration:
if args.verbalizer in ["kpt", "manual"]:
    if args.calibration or args.filter != "none":
        org_label_words_num = [len(prompt_model.verbalizer.label_words[i]) for i in range(len(class_labels))]
        from openprompt.utils.calibrate import calibrate

        # calculate the calibration logits
        cc_logits = calibrate(prompt_model, support_dataloader)
        print("the calibration logits is", cc_logits)
        print("origial label words num {}".format(org_label_words_num))

    if args.calibration:
        myverbalizer.register_calibrate_logits(cc_logits)
        new_label_words_num = [len(myverbalizer.label_words[i]) for i in range(len(class_labels))]
        print("After filtering, number of label words per class: {}".format(new_label_words_num))

    # register the logits to the verbalizer so that the verbalizer will divide the calibration probability in producing label logits
    # currently, only ManualVerbalizer and KnowledgeableVerbalizer support calibration.

from openprompt.data_utils.data_sampler import FewShotSampler

sampler = FewShotSampler(num_examples_per_label=args.shot, also_sample_dev=True, num_examples_per_label_dev=args.shot)
dataset['train'], dataset['validation'] = sampler(dataset['train'], seed=args.seed)
# print(dataset['train'])
# print(dataset['validation'])
# 第六步：构造PromptDateLoader，与数据加载和数据处理有关
#    PromptDataLoader基本上是pytorch DataLoader的prompt版本，它还包括一个Tokerizer、一个Template和一个TokenizeerWrapper
train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
                                    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                    decoder_max_length=3,
                                    batch_size=batch_s, shuffle=True, teacher_forcing=False, predict_eos_token=False,
                                    truncate_method="tail")
# print(train_dataloader.raw_dataset)
validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer,
                                         tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                         decoder_max_length=3,
                                         batch_size=batch_s, shuffle=False, teacher_forcing=False,
                                         predict_eos_token=False,
                                         truncate_method="tail")

# zero-shot test
test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
                                   tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3,
                                   batch_size=batch_s, shuffle=False, teacher_forcing=False, predict_eos_token=False,
                                   truncate_method="tail")


def evaluate1(prompt_model, dataloader, desc):
    prompt_model.eval()
    allpreds = []
    alllabels = []
    pbar = tqdm.tqdm(dataloader, desc=desc)
    for step, inputs in enumerate(pbar):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    # print(alllabels)
    # print(allpreds)

    # acc = sum([int(i == j) for i, j in zip(allpreds, alllabels)]) / len(allpreds)
    # Micro_F1 = metrics.classification_metrics(allpreds,alllabels,metric="micro-f1")
    acc = accuracy_score(alllabels,allpreds)
    pre = precision_score(alllabels,allpreds,average='weighted')
    recall = recall_score(alllabels,allpreds,average='weighted')
    F1_sc = f1_score(alllabels,allpreds,average='weighted')
    eval = [acc, pre, recall, F1_sc]
    return eval


def evaluate2(prompt_model, dataloader, desc):
    prompt_model.eval()
    allpreds = []
    alllabels = []
    pbar = tqdm.tqdm(dataloader, desc=desc)

    for step, inputs in enumerate(pbar):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        f_w = open('hc3_finance_prelabel_0.csv', 'w', encoding='utf-8')
        for i in range(len(allpreds)):
            x = allpreds[i]
            x = str(x)
            f_w.write(x + '\n')

    # print(alllabels)
    # print(allpreds)

    acc = sum([int(i == j) for i, j in zip(allpreds, alllabels)]) / len(allpreds)
    # Micro_F1 = metrics.classification_metrics(allpreds,alllabels,metric="micro-f1")
    return acc


############


from transformers import AdamW, get_linear_schedule_with_warmup

# 交叉熵损失：将Log_Softmax()激活函数与NLLLoss()损失函数的功能综合在一起
loss_func = torch.nn.CrossEntropyLoss()


def prompt_initialize(verbalizer, prompt_model, init_dataloader):
    dataloader = init_dataloader
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Init_using_{}".format("train")):
            batch = batch.cuda()
            logits = prompt_model(batch)
        verbalizer.optimize_to_initialize()


if args.verbalizer == "soft":

    no_decay = ['bias', 'LayerNorm.weight']

    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    # Using different optimizer for prompt parameters and model parameters

    optimizer_grouped_parameters2 = [
        {'params': prompt_model.verbalizer.group_parameters_1, "lr": 3e-5},
        {'params': prompt_model.verbalizer.group_parameters_2, "lr": 3e-4},
    ]

    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5)
    optimizer2 = AdamW(optimizer_grouped_parameters2)

    tot_step = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1,
        num_warmup_steps=0, num_training_steps=tot_step)

    scheduler2 = get_linear_schedule_with_warmup(
        optimizer2,
        num_warmup_steps=0, num_training_steps=tot_step)

elif args.verbalizer == "auto":
    prompt_initialize(myverbalizer, prompt_model, train_dataloader)

    no_decay = ['bias', 'LayerNorm.weight']

    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    # Using different optimizer for prompt parameters and model parameters

    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5)

    tot_step = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1,
        num_warmup_steps=0, num_training_steps=tot_step)

    optimizer2 = None
    scheduler2 = None

elif args.verbalizer == "kpt":
    no_decay = ['bias', 'LayerNorm.weight']

    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    # Using different optimizer for prompt parameters and model parameters

    # optimizer_grouped_parameters2 = [
    #     {'params': , "lr":1e-1},
    # ]
    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5)

    optimizer2 = AdamW(prompt_model.verbalizer.parameters(), lr=args.kptw_lr)
    # print(optimizer_grouped_parameters2)

    tot_step = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1,
        num_warmup_steps=0, num_training_steps=tot_step)

    # scheduler2 = get_linear_schedule_with_warmup(
    #     optimizer2,
    #     num_warmup_steps=0, num_training_steps=tot_step)
    scheduler2 = None

elif args.verbalizer == "manual":
    no_decay = ['bias', 'LayerNorm.weight']

    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    # Using different optimizer for prompt parameters and model parameters

    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5)

    tot_step = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1,
        num_warmup_steps=0, num_training_steps=tot_step)

    optimizer2 = None
    scheduler2 = None

#  第七步： 训练和测试

prompt_model.load_state_dict(torch.load(f"./ckpts/waimai-hotel_tem0.ckpt"))


test_acc = evaluate1(prompt_model, test_dataloader, desc="Test")



end_time = time.time()
run_time = end_time - start_time

content_write = "=" * 20 + "\n"
content_write += f"dataset {args.dataset}\t"
content_write += f"temp {args.template_id}\t"
content_write += f"seed {args.seed}\t"
content_write += f"shot {args.shot}\t"
content_write += f"verb {args.verbalizer}\t"
content_write += f"cali {args.calibration}\t"
content_write += f"filt {args.filter}\t"
content_write += f"maxsplit {args.max_token_split}\t"
content_write += f"kptw_lr {args.kptw_lr}\t"
content_write += f"b_t {args.batch_size}\t"
content_write += f"epoch {args.max_epochs}\t"
content_write += "\n"
content_write += f"run time {run_time}\t"
content_write += "\n"
content_write += f"Acc: {test_acc}"
content_write += "\n\n"

print(content_write)

with open(f"{args.result_file}", "a") as fout:
    fout.write(content_write)

# with open('time.csv', 'a+', encoding='utf-8') as t:
#     t.write('dataset: ' + args.dataset + ' template: ' + str(args.template_id) + ' 1次prompt time:' + str(run_time) + '\n')

