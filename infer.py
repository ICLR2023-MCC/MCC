from tf_env import tf
import numpy as np
from fake_data import FakeData
from CEN import CEN
from CS import CS
from MCCAN import MCCAN
from config import *

fake_data = FakeData(batch_size=Config.batch_size, lstm_step=Config.lstm_step, player_num=Config.player_num)

size_info = fake_data.get_size()
data = fake_data.get_data()
placeholder = fake_data.get_placeholder()
ph_data_dict = fake_data.get_placeholder_data_dict(placeholder, data)

# print(ph_data_dict)

### CEN ###
CEN_prob_list = []
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
        CEN_prob_list.append(softmax_prob)
        cmd_set.append(meta_cmd)
print(CEN_prob_list, cmd_set)

### CS ###
CS_value_list = []
CS_q_value_list = []
select_cmd_list = []
feature_list = ['hero', 'monster', 'turret', 'minion', 'stat', 'invisible']
hero_size, monster_size, turret_size, minion_size, stat_size, invisible_size = [size_info[k] for k in feature_list]
cs_net = CS(hero_size=hero_size, monster_size=monster_size, turret_size=turret_size, minion_size=minion_size,
            stat_size=stat_size, invisible_size=invisible_size, meta_cmd_size=size_info['meta_cmd'], value_head=Config.value_head)
cs_net.build(is_train=False)
for index, player in enumerate(placeholder):
    with tf.variable_scope('player_%d' % index):
        hero, monster, turret, minion, stat, invisible = [player[k] for k in feature_list]
        cs_value, meta_cmd_q_value, select_cmd = cs_net.infer(hero=hero, monster=monster, turret=turret, minion=minion, stat=stat,
                                                  invisible=invisible, cmd_set=cmd_set)
        CS_value_list.append(cs_value)
        CS_q_value_list.append(meta_cmd_q_value)
        select_cmd_list.append(select_cmd)
print(CS_value_list, CS_q_value_list, select_cmd_list)

### MCCAN ###
feature_list = ['spatial', 'hero', 'monster', 'turret', 'minion', 'stat', 'invisible']
spatial_size, hero_size, monster_size, turret_size, minion_size, stat_size, invisible_size = [size_info[k] for k in feature_list]
mccan_net = MCCAN(spatial_size=spatial_size, hero_size=hero_size, monster_size=monster_size, turret_size=turret_size, minion_size=minion_size,
                  stat_size=stat_size, invisible_size=invisible_size, meta_cmd_size=size_info['meta_cmd'])
mccan_net.build(is_train=False)
each_hero_result_list = mccan_net.infer(placeholder, select_cmd_list)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    result_val = sess.run(each_hero_result_list, feed_dict=ph_data_dict)
