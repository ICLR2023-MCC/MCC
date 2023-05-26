from tf_env import tf
import numpy as np
from fake_data import FakeData
from CEN import CEN
from MCCAN import MCCAN
from config import *

fake_data = FakeData(batch_size=Config.batch_size, lstm_step=Config.lstm_step, player_num=Config.player_num)

size_info = fake_data.get_size()
data = fake_data.get_data()
placeholder = fake_data.get_placeholder()
ph_data_dict = fake_data.get_placeholder_data_dict(placeholder, data)

### CEN ###
self_cmd_list = []
feature_list = ['hero', 'monster', 'turret', 'minion', 'stat']
hero_size, monster_size, turret_size, minion_size, stat_size = [size_info[k] for k in feature_list]
cen_net = CEN(hero_size=hero_size, monster_size=monster_size, turret_size=turret_size, minion_size=minion_size,
              stat_size=stat_size, meta_cmd_size=size_info['meta_cmd'], meta_cmd_E_size=size_info['meta_cmd_E'])
cen_net.build(is_train=False)
for index, player in enumerate(placeholder):
    with tf.variable_scope('player_%d' % index):
        hero, monster, turret, minion, stat = [player[k] for k in feature_list]
        softmax_prob, _, meta_cmd = cen_net.infer(hero=hero, monster=monster, turret=turret, minion=minion, stat=stat, top_k=CEN_config.softmax_k)
        self_cmd_list.append(meta_cmd)

### MCCAN ###
feature_list = ['spatial', 'hero', 'monster', 'turret', 'minion', 'stat', 'invisible']
spatial_size, hero_size, monster_size, turret_size, minion_size, stat_size, invisible_size = [size_info[k] for k in feature_list]
mccan_net = MCCAN(spatial_size=spatial_size, hero_size=hero_size, monster_size=monster_size, turret_size=turret_size, minion_size=minion_size,
                  stat_size=stat_size, invisible_size=invisible_size, meta_cmd_size=size_info['meta_cmd'])
mccan_net.build(is_train=True)
player_action_list, player_value_list = mccan_net.infer(player_feature_list=placeholder, select_cmd_list=self_cmd_list)

losses = []
for index, (player_action_policy, player_value) in enumerate(zip(player_action_list, player_value_list)):
    player_loss = tf.constant(0.0, dtype=tf.float32)

    V_label, advantage = placeholder[index]['MCCAN_V_label'], placeholder[index]['MCCAN_advantage']
    value_loss = 0.5 * tf.reduce_mean(tf.square(V_label - player_value), axis=0)
    player_loss += tf.reduce_sum(value_loss)

    for action_index, policy in enumerate(player_action_policy):
        old_policy, action_label = placeholder[index]['MCCAN_old_policy_%d' % action_index], placeholder[index]['MCCAN_action_label_%d' % action_index]

        policy_p = tf.reduce_sum(action_label * policy, axis=1)
        policy_log_p = tf.log(policy_p)
        old_policy_p = tf.reduce_sum(action_label * old_policy, axis=1)
        old_policy_log_p = tf.log(old_policy_p)
        ratio = tf.exp(policy_log_p - old_policy_log_p)
        surr1 = tf.clip_by_value(ratio, 0.0, MCCAN_config.dual_clip_param) * advantage
        surr2 = tf.clip_by_value(ratio, 1.0 - MCCAN_config.clip_param, 1.0 + MCCAN_config.clip_param) * advantage
        dual_ppo_loss = - tf.reduce_mean(tf.minimum(surr1, surr2))
        player_loss += dual_ppo_loss

        entropy_loss = -tf.reduce_mean(policy_p * policy_log_p)
        player_loss += entropy_loss

    losses.append(player_loss)
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