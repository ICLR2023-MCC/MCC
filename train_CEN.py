from tf_env import tf
import numpy as np
from fake_data import FakeData
from CEN import CEN
from config import *
from utils import focal_loss

fake_data = FakeData(batch_size=Config.batch_size, lstm_step=Config.lstm_step, player_num=Config.player_num)

size_info = fake_data.get_size()
data = fake_data.get_data()
placeholder = fake_data.get_placeholder()
ph_data_dict = fake_data.get_placeholder_data_dict(placeholder, data)

### infer ###
losses = []
feature_list = ['hero', 'monster', 'turret', 'minion', 'stat']
hero_size, monster_size, turret_size, minion_size, stat_size = [size_info[k] for k in feature_list]
cen_net = CEN(hero_size=hero_size, monster_size=monster_size, turret_size=turret_size, minion_size=minion_size,
              stat_size=stat_size, meta_cmd_size=size_info['meta_cmd'])
cen_net.build(is_train=True)
for index, player in enumerate(placeholder):
    with tf.variable_scope('player_%d' % index):
        hero, monster, turret, minion, stat = [player[k] for k in feature_list]
        y_label = player['CEN_label']
        softmax_prob, meta_cmd = cen_net.infer(hero=hero, monster=monster, turret=turret, minion=minion, stat=stat, top_k=CEN_config.softmax_k)
        loss = focal_loss(y_label, softmax_prob, alpha=0.75, gamma=2)
        losses.append(loss)
total_loss = tf.reduce_mean(losses)

### backpropagation ###
params = tf.trainable_variables()
optimizer = tf.train.AdamOptimizer(CEN_config.init_learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08)
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
