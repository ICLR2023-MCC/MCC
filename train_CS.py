from tf_env import tf
import numpy as np
from fake_data import FakeData
from CS import CS
from CEN import CEN
from config import *

fake_data = FakeData(batch_size=Config.batch_size, lstm_step=Config.lstm_step, player_num=Config.player_num)

size_info = fake_data.get_size()
data = fake_data.get_data()
placeholder = fake_data.get_placeholder()
ph_data_dict = fake_data.get_placeholder_data_dict(placeholder, data)

### CEN ###
cmd_set = []
feature_list = ['hero', 'monster', 'turret', 'minion', 'stat']
hero_size, monster_size, turret_size, minion_size, stat_size = [size_info[k] for k in feature_list]
cen_net = CEN(hero_size=hero_size, monster_size=monster_size, turret_size=turret_size, minion_size=minion_size,
              stat_size=stat_size, meta_cmd_size=size_info['meta_cmd'])
cen_net.build(is_train=False)
for index, player in enumerate(placeholder):
    with tf.variable_scope('player_%d' % index):
        hero, monster, turret, minion, stat = [player[k] for k in feature_list]
        softmax_prob, meta_cmd = cen_net.infer(hero=hero, monster=monster, turret=turret, minion=minion, stat=stat, top_k=CEN_config.softmax_k)
        cmd_set.append(meta_cmd)

### CS ###
losses = []
feature_list = ['hero', 'monster', 'turret', 'minion', 'stat', 'invisible']
hero_size, monster_size, turret_size, minion_size, stat_size, invisible_size = [size_info[k] for k in feature_list]
cs_net = CS(hero_size=hero_size, monster_size=monster_size, turret_size=turret_size, minion_size=minion_size,
            stat_size=stat_size, invisible_size=invisible_size, meta_cmd_size=size_info['meta_cmd'], value_head=Config.value_head)
cs_net.build(is_train=True)
for index, player in enumerate(placeholder):
    with tf.variable_scope('player_%d' % index):
        hero, monster, turret, minion, stat, invisible = [player[k] for k in feature_list]
        cs_value, meta_cmd_q_value, select_cmd = cs_net.infer(hero=hero, monster=monster, turret=turret, minion=minion, stat=stat,
                                                  invisible=invisible, cmd_set=cmd_set)
        V_label, Q_label, cmd_label = player['CS_V_label'], player['CS_Q_label'], player['CS_cmd_label']
        value_loss = 0.5 * tf.reduce_mean(tf.square(V_label - cs_value), axis=0)

        cmd_real_q = tf.reduce_sum(meta_cmd_q_value * cmd_label, axis=1)
        q_value_loss = 0.5 * tf.reduce_mean(tf.square(cmd_real_q - Q_label))

        one_player_loss = tf.reduce_sum(value_loss) + q_value_loss
        losses.append(one_player_loss)
total_loss = tf.reduce_sum(losses)

### backpropagation ###
params = tf.trainable_variables()
optimizer = tf.train.AdamOptimizer(CS_config.init_learning_rate, beta1=0.9, beta2=0.999, epsilon=0.00001)
grads = tf.gradients(total_loss, params)
train_op = optimizer.apply_gradients(zip(grads, params))

init = tf.global_variables_initializer()
global_step = tf.Variable(0, name='global_step', trainable=False)
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(CEN_config.epoch):
        result = sess.run([train_op], feed_dict=ph_data_dict)
        visual_loss = sess.run(total_loss, feed_dict=ph_data_dict)
        print('epoch %d: loss=%f' % (epoch, visual_loss))