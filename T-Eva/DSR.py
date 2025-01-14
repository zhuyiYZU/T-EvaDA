from __future__ import print_function

from T_Eva.Office_home.flip_gradient import flip_gradient
from T_Eva.Office_home.utils import *
from tensorflow.python.tools import inspect_checkpoint as chkp
from sklearn.metrics import precision_score,recall_score,f1_score
INPUT_DIM = 2048

from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.python.training import checkpoint_utils

class DSR:

    def __init__(self, input_feature_size, source_file, target_file, newfile, learning_rate, batch_size, num_steps,NUM_CLASS):
        self.input_shape = input_feature_size
        self.source_file = source_file
        self.target_file = target_file
        self.newfile = newfile
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.NUM_CLASS = NUM_CLASS
        pass

    def load_text_data(self,source_file, target_file):
        source_data = pd.read_csv(source_file)
        target_data = pd.read_csv(target_file)
        source_data = source_data.dropna(subset=['text'])
        target_data = target_data.dropna(subset=['text'])
        global vectorizer
        vectorizer = TfidfVectorizer(max_features=INPUT_DIM)
        source_features = vectorizer.fit_transform(source_data['text']).toarray()
        target_features = vectorizer.transform(target_data['text']).toarray()

        source_labels = source_data['label'].values
        target_labels = target_data['label'].values

        return source_features, source_labels, target_features, target_labels, target_features, target_labels
    def inference(self, x, is_reuse, l, is_training, dropout_rate=0.5):

        semantic_label_logits, semantic_domain_logits, semantic_latent_var, semantic_mean, semantic_stddev = \
            self.semantic_vae(feature=x, is_training=is_training, is_reuse=is_reuse, l=l)

        domain_label_logits, domain_domain_logits, domain_latent_var, domain_mean, domain_stddev = self.domain_vae(
            feature=x, is_training=is_training, is_reuse=is_reuse, l=l)

        total_latent_var = tf.concat([semantic_latent_var, domain_latent_var], axis=-1)
        total_latent_var = tf.contrib.layers.layer_norm(total_latent_var, scale=True)

        reconstruct_input = self.decoder(latent_var=total_latent_var, is_reuse=is_reuse, is_training=is_training,
                                         dropout_rate=dropout_rate)

        total_mean = tf.concat([semantic_mean, domain_mean], axis=-1)
        total_stddev = tf.concat([semantic_stddev, domain_stddev], axis=-1)

        result = dict()

        result["reconstruct_input"] = reconstruct_input

        result["semantic_label_logits"] = semantic_label_logits
        result["semantic_domain_logits"] = semantic_domain_logits
        result["semantic_mean"] = semantic_mean
        result["semantic_stddev"] = semantic_stddev

        result["domain_label_logits"] = domain_label_logits
        result["domain_domain_logits"] = domain_domain_logits
        result["domain_mean"] = domain_mean
        result["domain_stddev"] = domain_stddev

        result["total_mean"] = total_mean
        result["total_stddev"] = total_stddev

        return result

    def decoder (self, latent_var, is_reuse, is_training, dropout_rate):
        with tf.variable_scope("feature_decoder", reuse=is_reuse):
            decoder_w0 = weight_variable(shape=[4000, self.input_shape], name="decoder_weight_0")
            decoder_b0 = bias_variable(shape=[self.input_shape], name="decoder_biases_0")
            decoder_h0 = tf.matmul(latent_var, decoder_w0) + decoder_b0
            decoder_h0 = tf.nn.relu(decoder_h0)
            decoder_h0 = tf.layers.dropout(decoder_h0, training=is_training, rate=dropout_rate)
            decoder_h0 = tf.contrib.layers.layer_norm(decoder_h0, scale=True)

        return decoder_h0

    def semantic_vae(self, feature, is_reuse, is_training, l):
        with tf.variable_scope("semantic_vae", reuse=is_reuse):
            encoder_output_mu_w = weight_variable(shape=[self.input_shape, 2000], name="encoder_output_mu_w")
            encoder_output_mu_b = bias_variable(shape=[2000], name="encoder_output_mu_b")
            mean = tf.matmul(feature, encoder_output_mu_w) + encoder_output_mu_b
            mean = tf.contrib.layers.batch_norm(mean, scale=True)

            encoder_output_stddev_w = weight_variable(shape=[self.input_shape, 2000], name="encoder_output_stddev_w")
            encoder_output_stddev_b = bias_variable(shape=[2000], name="encoder_output_stddev_b")
            stddev = tf.matmul(feature, encoder_output_stddev_w) + encoder_output_stddev_b
            stddev = tf.contrib.layers.batch_norm(stddev, scale=True)

            semantic_latent_var = self.reparamized(mu=mean, sigma=stddev, is_training=is_training)

        with tf.variable_scope("semantic_label_predictor", reuse=is_reuse):
            W_fc0 = weight_variable(shape=[2000, 3000], name="fc0_Weight")
            h_fc0 = tf.nn.relu(tf.matmul(semantic_latent_var, W_fc0) + 0)
            h_fc0 = tf.layers.dropout(h_fc0, training=is_training)
            h_fc0 = tf.contrib.layers.batch_norm(h_fc0, scale=True)

            W_fc1 = weight_variable(shape=[3000, self.NUM_CLASS], name="fc1_Weight")
            b_fc1 = bias_variable(shape=[self.NUM_CLASS], name="fc1_b")
            label_logits = tf.matmul(h_fc0, W_fc1) + b_fc1

        with tf.variable_scope("semantic_domain_predictor", reuse=is_reuse):
            feat = flip_gradient(semantic_latent_var, l)
            d_W_fc0 = weight_variable(shape=[2000, 400], name="fcd_w0")
            d_b_fc0 = bias_variable(shape=[400], name="fcd_b0")
            d_h_fc0 = tf.nn.relu(tf.matmul(feat, d_W_fc0) + d_b_fc0)

            d_h_fc0 = tf.layers.dropout(d_h_fc0, training=is_training, rate=0.90)
            d_h_fc0 = tf.contrib.layers.batch_norm(d_h_fc0, scale=True)

            d_W_fc1 = weight_variable(shape=[400, 2], name="fcd_w1")
            d_b_fc1 = bias_variable(shape=[2], name="fcd_b1")
            domain_logits = tf.matmul(d_h_fc0, d_W_fc1) + d_b_fc1

        return label_logits, domain_logits, semantic_latent_var, mean, stddev

    def domain_vae (self, feature, is_reuse, is_training, l):
        with tf.variable_scope("domain_vae", reuse=is_reuse):
            encoder_output_mu_w = weight_variable(shape=[self.input_shape, 2000], name="encoder_output_mu_w")
            encoder_output_mu_b = bias_variable(shape=[2000], name="encoder_output_mu_b")
            mean = tf.matmul(feature, encoder_output_mu_w) + encoder_output_mu_b
            mean = tf.contrib.layers.batch_norm(mean, scale=True)

            encoder_output_stddev_w = weight_variable(shape=[self.input_shape, 2000],
                                                          name="encoder_output_stddev_w")
            encoder_output_stddev_b = bias_variable(shape=[2000], name="encoder_output_stddev_b")
            stddev = tf.matmul(feature, encoder_output_stddev_w) + encoder_output_stddev_b
            stddev = tf.contrib.layers.batch_norm(stddev, scale=True)

            domain_latent_var = self.reparamized(mu=mean, sigma=stddev, is_training=is_training)

        with tf.variable_scope("domain_label_predictor", reuse=is_reuse):
            feat = flip_gradient(domain_latent_var, l)

            W_fc0 = weight_variable(shape=[2000, 2000], name="fc0_Weight")
            b_fc0 = bias_variable(shape=[2000], name="fc0_b")
            h_fc0 = tf.nn.relu(tf.matmul(feat, W_fc0) + b_fc0)

            h_fc0 = tf.layers.dropout(h_fc0, training=is_training)
            h_fc0 = tf.contrib.layers.batch_norm(h_fc0, scale=True)

            W_fc1 = weight_variable(shape=[2000, self.NUM_CLASS], name="fc1_Weight")
            b_fc1 = bias_variable(shape=[self.NUM_CLASS], name="fc1_b")
            label_logits = tf.matmul(h_fc0, W_fc1) + b_fc1

        with tf.variable_scope("domain_domain_predictor", reuse=is_reuse):
            d_W_fc0 = weight_variable(shape=[2000, 200], name="fcd_w0")
            d_b_fc0 = bias_variable(shape=[200], name="fcd_b0")
            d_h_fc0 = tf.nn.relu(tf.matmul(domain_latent_var, d_W_fc0) + d_b_fc0)

            d_h_fc0 = tf.layers.dropout(d_h_fc0, training=is_training)
            d_h_fc0 = tf.contrib.layers.batch_norm(d_h_fc0, scale=True)

            d_W_fc1 = weight_variable(shape=[200, 2], name="fcd_w1")
            d_b_fc1 = bias_variable(shape=[2], name="fcd_b1")
            domain_logits = tf.matmul(d_h_fc0, d_W_fc1) + d_b_fc1

        return label_logits, domain_logits, domain_latent_var, mean, stddev

    def reparamized (self, mu, sigma, is_training):
        std = tf.exp(0.5 * sigma)
        eps = tf.random_normal(tf.shape(std), 0, 1, dtype=tf.float32)

        z = tf.cond(is_training, lambda: eps * std + mu, lambda: mu)

        return z

    def run(self):
        source_name = 'source'
        target_name = 'target'
        source_FILE = self.source_file
        target_FILE = self.target_file

        source_train_input, source_train_label, target_train_input, target_train_label, target_test_input, target_test_label = self.load_text_data(
            source_file=source_FILE, target_file=target_FILE)

        source_train_y = get_one_hot_label(source_train_label)
        target_train_y = get_one_hot_label(target_train_label)
        target_test_y = get_one_hot_label(target_test_label)

        # 加载目标域 20% 测试数据
        newtest_FILE = self.newfile

        newtest_data = pd.read_csv(newtest_FILE)
        newtest_data = newtest_data.dropna(subset=['text'])

        # 使用与训练一致的 TF-IDF 向量化器进行特征提取
        newtest_features = vectorizer.transform(newtest_data['text']).toarray()
        newtest_labels = newtest_data['label'].values
        newtest_y = get_one_hot_label(newtest_labels)

        print(source_train_input.shape)
        print(source_train_label.shape)
        print(target_train_input.shape)
        print(target_train_label.shape)
        print(target_test_input.shape)
        print(target_test_label.shape)

        batch_size = 128
        num_steps = 100000  # 训练轮次

        graph = tf.get_default_graph()
        with graph.as_default():
            model = DSR(input_feature_size=INPUT_DIM, source_file=self.source_file, target_file=self.target_file, newfile=self.newfile, learning_rate=self.learning_rate, batch_size=self.batch_size, num_steps=self.num_steps,NUM_CLASS=self.NUM_CLASS)

            source_input = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_DIM], name="source_input")
            source_label = tf.placeholder(dtype=tf.int32, shape=[None, self.NUM_CLASS])
            target_input = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_DIM])
            target_label = tf.placeholder(dtype=tf.int32, shape=[None, self.NUM_CLASS])

            source_domain = tf.placeholder(tf.float32, [None, 2])
            target_domain = tf.placeholder(tf.float32, [None, 2])
            learning_rate = tf.placeholder(tf.float32, [])

            alpha = tf.placeholder(tf.float32, [])
            gama = tf.placeholder(tf.float32, [])
            train_mode = tf.placeholder(tf.bool, [], name="train_mode")

            drop_rate = 0.5
            beta = 150
            source_result = model.inference(x=source_input, is_reuse=False, is_training=train_mode, l=alpha,
                                            dropout_rate=drop_rate)
            target_result = model.inference(x=target_input, is_reuse=True, is_training=train_mode, l=alpha,
                                            dropout_rate=drop_rate)

            # source vae loss
            src_reconstruct_loss = tf.sqrt(
                tf.reduce_sum(tf.square(source_input - source_result["reconstruct_input"]), axis=-1))
            src_reconstruct_loss = tf.reduce_mean(src_reconstruct_loss)

            src_KL_divergence = 0.5 * tf.reduce_mean(
                1 + source_result["total_stddev"] - tf.square(source_result["total_mean"]) - tf.exp(
                    source_result["total_stddev"]))

            source_ELOB = src_reconstruct_loss - beta * src_KL_divergence

            source_vae_loss = source_ELOB

            # target vae loss
            tgt_reconstruct_loss = tf.sqrt(
                tf.reduce_sum(tf.square(target_input - target_result["reconstruct_input"]), axis=-1))
            tgt_reconstruct_loss = tf.reduce_mean(tgt_reconstruct_loss)

            tgt_KL_divergence = 0.5 * tf.reduce_mean(
                1 + target_result["total_stddev"] - tf.square(target_result["total_mean"]) - tf.exp(
                    target_result["total_stddev"]))

            target_ELOB = tgt_reconstruct_loss - beta * tgt_KL_divergence

            target_vae_loss = target_ELOB

            # total vae loss
            vae_loss = (source_vae_loss + target_vae_loss) / 2

            # semantic label loss
            src_semantic_label_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=source_result["semantic_label_logits"],
                                                           labels=source_label))

            semantic_label_loss = src_semantic_label_loss

            # semantic domain loss
            semantic_domain_logits = tf.concat(
                [source_result["semantic_domain_logits"], target_result["semantic_domain_logits"]], axis=0)
            source_target_domain_label = tf.concat([source_domain, target_domain], axis=0)
            semantic_domain_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=semantic_domain_logits,
                                                           labels=source_target_domain_label))

            src_predict = tf.nn.softmax(source_result["domain_label_logits"], axis=-1)
            tgt_predict = tf.nn.softmax(target_result["domain_label_logits"], axis=-1)

            """src_entropy and tgt_entropy is the negative entropy, '2+ ' make them be a positive value"""
            src_entropy = 2 + tf.reduce_mean(tf.reduce_sum(tf.log(src_predict) * src_predict, axis=-1))
            tgt_entropy = 2 + tf.reduce_mean(tf.reduce_sum(tf.log(tgt_predict) * tgt_predict, axis=-1))

            domain_label_loss = (src_entropy + tgt_entropy) / 2

            # domain domain loss
            domain_domain_logits = tf.concat(
                [source_result["domain_domain_logits"], target_result["domain_domain_logits"]],
                0)
            domain_domain_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=domain_domain_logits,
                                                           labels=source_target_domain_label))

            total_loss = gama * vae_loss + (
                    semantic_label_loss + semantic_domain_loss + domain_domain_loss + domain_label_loss)

            for param in tf.trainable_variables():
                print(param)

            weight_para = [v for v in tf.trainable_variables() if "Weight" in v.name]

            bias_para = [v for v in tf.trainable_variables() if "Weight" not in v.name]

            label_b_para = [v for v in tf.trainable_variables() if "b" in v.name and "label" not in v.name]

            greg_loss = 5e-3 * tf.reduce_mean([tf.nn.l2_loss(x) for x in bias_para])
            biases_greg_loss = 5e-5 * tf.reduce_mean([tf.nn.l2_loss(x) for x in label_b_para])
            predictor_greg_loss = 5e-4 * tf.reduce_mean([tf.nn.l2_loss(x) for x in weight_para])

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):

                opt_1 = tf.train.MomentumOptimizer(learning_rate, 0.50)
                weight_gradients = tf.gradients(4.0 * (total_loss + greg_loss), bias_para)
                dann_train_op_1 = opt_1.apply_gradients(zip(weight_gradients, bias_para))

                opt_2 = tf.train.MomentumOptimizer(learning_rate, 0.5)
                bias_gradients = tf.gradients(1.0 * (total_loss + predictor_greg_loss), weight_para)
                bias_clipped_gradients, _ = tf.clip_by_global_norm(bias_gradients, 0.15)
                dann_train_op_2 = opt_2.apply_gradients(zip(bias_clipped_gradients, weight_para))

                opt_3 = tf.train.MomentumOptimizer(learning_rate, 0.50)
                b_weight_gradients = tf.gradients(1.0 * (total_loss + biases_greg_loss), label_b_para)
                dann_train_op_3 = opt_3.apply_gradients(zip(b_weight_gradients, label_b_para))

                dann_train_op = tf.group([dann_train_op_1, dann_train_op_2, dann_train_op_3])

            target_correct_label = tf.equal(tf.argmax(target_label, 1),
                                            tf.argmax(target_result["semantic_label_logits"], 1))
            target_label_acc = tf.reduce_mean(tf.cast(target_correct_label, tf.float32))

            saver = tf.train.Saver(tf.trainable_variables())

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

            with tf.Session(config=config) as session:
                # 初始化所有变量
                session.run(tf.global_variables_initializer())

                # 生成训练数据的批次
                gen_source_batch = batch_generator([source_train_input, source_train_y], batch_size)
                gen_target_batch = batch_generator([target_train_input, target_train_y], batch_size)

                source_domain_input = np.tile([1., 0.], [batch_size, 1])
                target_domain_input = np.tile([0., 1.], [batch_size, 1])

                init_L = 0.15
                best_result = 0.0
                lr = self.learning_rate
                g = 1.0

                for global_steps in range(num_steps):
                    p = float(global_steps) / num_steps
                    L = init_L

                    # 获取源域和目标域批次数据
                    X0, y0 = next(gen_source_batch)
                    X1, y1 = next(gen_target_batch)

                    # 执行一次训练步
                    _, batch_total_loss, batch_vae_loss, batch_semantic_label_loss, batch_semantic_domain_loss, batch_label_loss, batch_domain_domain_loss = session.run(
                        [dann_train_op, total_loss, vae_loss, semantic_label_loss, semantic_domain_loss,
                         domain_label_loss,
                         domain_domain_loss],
                        feed_dict={source_input: X0, source_label: y0, target_input: X1, learning_rate: lr, alpha: L,
                                   source_domain: source_domain_input, target_domain: target_domain_input,
                                   train_mode: True,
                                   gama: g})

                    # 每 200 步评估一次 `newtest` 的准确率
                    if global_steps % 200 == 0:
                        target_label_accuracy = session.run(target_label_acc,
                                                            feed_dict={target_input: target_test_input,
                                                                       target_label: target_test_y,
                                                                       train_mode: False})
                        # 新增：评估 Precision、Recall 和 F1
                        test_predictions = session.run(tf.argmax(target_result["semantic_label_logits"], axis=1),
                                                       feed_dict={target_input: target_test_input, train_mode: False})
                        test_true_labels = np.argmax(target_test_y, axis=1)

                        test_precision = precision_score(test_true_labels, test_predictions, average='macro')
                        test_recall = recall_score(test_true_labels, test_predictions, average='macro')
                        test_f1 = f1_score(test_true_labels, test_predictions, average='macro')

                        # 评估 `newtest` 的准确率
                        newtest_predictions = session.run(tf.argmax(target_result["semantic_label_logits"], axis=1),
                                                          feed_dict={target_input: newtest_features, train_mode: False})
                        newtest_accuracy = np.mean(newtest_predictions == newtest_labels)
                        # 新增：评估 `newtest` 的 Precision、Recall 和 F1
                        newtest_precision = precision_score(newtest_labels, newtest_predictions, average='macro')
                        newtest_recall = recall_score(newtest_labels, newtest_predictions, average='macro')
                        newtest_f1 = f1_score(newtest_labels, newtest_predictions, average='macro')

                        # 保存最佳模型
                        if target_label_accuracy > best_result:
                            saver.save(session, "model_result/%s-%s.ckpt" % (source_name, target_name))
                            best_result = target_label_accuracy

                        # 打印结果
                        print(f"Global Step: {global_steps}")
                        print(f"Batch Total Loss: {batch_total_loss}")
                        print(f"Batch VAE Loss: {batch_vae_loss}")
                        print(f"Batch Semantic Label Loss: {batch_semantic_label_loss}")
                        print(f"Batch Semantic Domain Loss: {batch_semantic_domain_loss}")
                        print(f"Batch Domain Label Loss: {batch_label_loss}")
                        print(f"Batch Domain Domain Loss: {batch_domain_domain_loss}")
                        print(f"Target Label Accuracy: {target_label_accuracy:.4f}")
                        print(f"Newtest Accuracy: {newtest_accuracy:.4f}")
                        print(f"Best Result: {best_result:.4f}")

                        # 新增 Precision、Recall 和 F1 指标打印
                        print(
                            f"Ori. Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}")
                        print(
                            f"T-Eva Precision: {newtest_precision:.4f}, Newtest Recall: {newtest_recall:.4f}, Newtest F1: {newtest_f1:.4f}")
                        print(f"Best Result: {best_result:.4f}")


def get_one_hot_label(y):
    n_values = np.max(y) + 1
    one_hot_label = np.eye(n_values)[y]
    return one_hot_label







