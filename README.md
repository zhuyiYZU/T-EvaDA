**Datasets:**

In the T-EvaDA package, four widely recognized domain
adaptation datasets are integrated. These datasets include
English datasets such as the Amazon dataset and 20 Newsgroups dataset; Chinese datasets such as the Sentiment
Analysis dataset and ChineseNlpCorpus dataset. The details
of these datasets are as follows.

_Amazon dataset_
: The dataset is a benchmark for sentiment analysis of English product reviews. It contains reviews across four categories: Books (B), DVDs (D), Electronics (E), and Kitchen appliances (K). Reviews with 1 or 2
stars are labeled as negative, while those with 4 or 5 stars
are labeled as positive.

_20 Newsgroups dataset_
: The dataset is an English topic classification dataset comprising 18,774 news articles
organized hierarchically into six main categories and 20
subcategories. It contains four main categories including
Computer(Comp), Recreation(Rec), Science(Sci), and Talk. It
is worth noting that the distribution of source and target
domains differs due to the different subcategories.

_Sentiment Analysis dataset_
: The dataset contains Chinese reviews classified as positive or negative across four
domains: phone, camera, notebook, and car. This binary
classification dataset is suitable for cross-domain sentiment
analysis. In the package, we used 586 labeled car reviews
and 1116 labeled camera reviews as the source domain. The
target domain comprised 311 unlabeled notebook reviews,
586 unlabeled car reviews, 1115 unlabeled camera reviews,
and 1277 unlabeled phone reviews.

_Chinese NlpCorpus dataset_
: The Chinese dataset includes
reviews from various domains. In the package, we selected
three domains: hotel, waimai, and weibo, all of which include positive and negative reviews. The hotel dataset comprises 2000 labeled and unlabeled samples, while waimai
and weibo each contain 4000 samples. Labeled data were
used as the source domain, and unlabeled data were used
as the target domain

We have uploaded the corresponding dataset to the cloud storage:[click here](https://pan.baidu.com/s/1mZgaeGJwmPE92ZvuGa0f1A?pwd=e7w4
)

**Usage Guide:**
You can use the program under T-Eva/test to test these 7 methods, while also paying attention to the following points.

1.Different methods correspond to different dataset formats, please ensure the correctness of the format first.

2.Before using the Bert, Spot, and ItSPT methods, it is necessary to download the basic model.

3.Each method has some parameters that can be adjusted appropriately.

4.Finally, download the required environment for the program according to the requirement. txt.