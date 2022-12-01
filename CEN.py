from tf_env import tf
from utils import _fc_weight_variable, _bias_variable, maxpooling, topk_softmax, sample_from_prob
import numpy as np
from config import CEN_config

class CEN:
    def __init__(self, hero_size, monster_size, turret_size, minion_size, stat_size, meta_cmd_size, meta_cmd_E_size):
        self.hero_size = hero_size
        self.monster_size = monster_size
        self.turret_size = turret_size
        self.minion_size = minion_size
        self.stat_size = stat_size

        self.meta_cmd_size = meta_cmd_size
        self.meta_cmd_E_size = meta_cmd_E_size

        self.hero_dim = CEN_config.hero_dim
        self.unit_dim = CEN_config.unit_dim
        self.stat_dim = CEN_config.stat_dim
        self.meta_cmd_dim = CEN_config.meta_cmd_dim

    def build(self, is_train):
        with tf.variable_scope('CEN'):
            hero_flatten_size = int(np.prod(self.hero_size))
            self.fc1_hero_weight = _fc_weight_variable(shape=[hero_flatten_size, self.hero_dim], name="fc1_hero_weight", trainable=is_train)
            self.fc1_hero_bias = _bias_variable(shape=[self.hero_dim], name="fc1_hero_bias", trainable=is_train)
            self.fc2_hero_weight = _fc_weight_variable(shape=[self.hero_dim, self.hero_dim // 2], name="fc2_hero_weight", trainable=is_train)
            self.fc2_hero_bias = _bias_variable(shape=[self.hero_dim // 2], name="fc2_hero_bias", trainable=is_train)
            self.fc3_hero_weight = _fc_weight_variable(shape=[self.hero_dim // 2, self.hero_dim // 4], name="fc3_hero_weight", trainable=is_train)
            self.fc3_hero_bias = _bias_variable(shape=[self.hero_dim // 4], name="fc3_hero_bias", trainable=is_train)

            monster_flatten_size = int(np.prod(self.monster_size))
            self.fc1_monster_weight = _fc_weight_variable(shape=[monster_flatten_size, self.unit_dim], name="fc1_monster_weight", trainable=is_train)
            self.fc1_monster_bias = _bias_variable(shape=[self.unit_dim], name="fc1_monster_bias", trainable=is_train)
            self.fc2_monster_weight = _fc_weight_variable(shape=[self.unit_dim, self.unit_dim // 2], name="fc2_monster_weight", trainable=is_train)
            self.fc2_monster_bias = _bias_variable(shape=[self.unit_dim // 2], name="fc2_monster_bias", trainable=is_train)
            self.fc3_monster_weight = _fc_weight_variable(shape=[self.unit_dim // 2, self.unit_dim // 4], name="fc3_monster_weight", trainable=is_train)
            self.fc3_monster_bias = _bias_variable(shape=[self.unit_dim // 4], name="fc3_monster_bias", trainable=is_train)

            turret_flatten_size = int(np.prod(self.turret_size))
            self.fc1_turret_weight = _fc_weight_variable(shape=[turret_flatten_size, self.unit_dim], name="fc1_turret_weight", trainable=is_train)
            self.fc1_turret_bias = _bias_variable(shape=[self.unit_dim], name="fc1_turret_bias", trainable=is_train)
            self.fc2_turret_weight = _fc_weight_variable(shape=[self.unit_dim, self.unit_dim // 2], name="fc2_turret_weight", trainable=is_train)
            self.fc2_turret_bias = _bias_variable(shape=[self.unit_dim // 2], name="fc2_turret_bias", trainable=is_train)
            self.fc3_turret_weight = _fc_weight_variable(shape=[self.unit_dim // 2, self.unit_dim // 4], name="fc3_turret_weight", trainable=is_train)
            self.fc3_turret_bias = _bias_variable(shape=[self.unit_dim // 4], name="fc3_turret_bias", trainable=is_train)

            minion_flatten_size = int(np.prod(self.minion_size))
            self.fc1_minion_weight = _fc_weight_variable(shape=[minion_flatten_size, self.unit_dim], name="fc1_minion_weight", trainable=is_train)
            self.fc1_minion_bias = _bias_variable(shape=[self.unit_dim], name="fc1_minion_bias", trainable=is_train)
            self.fc2_minion_weight = _fc_weight_variable(shape=[self.unit_dim, self.unit_dim // 2], name="fc2_minion_weight", trainable=is_train)
            self.fc2_minion_bias = _bias_variable(shape=[self.unit_dim // 2], name="fc2_minion_bias", trainable=is_train)
            self.fc3_minion_weight = _fc_weight_variable(shape=[self.unit_dim // 2, self.unit_dim // 4], name="fc3_minion_weight", trainable=is_train)
            self.fc3_minion_bias = _bias_variable(shape=[self.unit_dim // 4], name="fc3_minion_bias", trainable=is_train)

            stat_flatten_size = int(np.prod(self.stat_size))
            self.fc1_stat_weight = _fc_weight_variable(shape=[stat_flatten_size, self.stat_dim], name="fc1_stat_weight", trainable=is_train)
            self.fc1_stat_bias = _bias_variable(shape=[self.stat_dim], name="fc1_stat_bias", trainable=is_train)
            self.fc2_stat_weight = _fc_weight_variable(shape=[self.stat_dim, self.stat_dim // 4], name="fc2_stat_weight", trainable=is_train)
            self.fc2_stat_bias = _bias_variable(shape=[self.stat_dim // 4], name="fc2_stat_bias", trainable=is_train)

            concat_dim = self.hero_dim // 4 + self.unit_dim // 4 * 3 + self.stat_dim // 4
            meta_cmd_flatten_size = int(np.prod(self.meta_cmd_size))
            self.fc1_meta_cmd_weight = _fc_weight_variable(shape=[concat_dim, self.meta_cmd_dim], name="fc1_meta_cmd_weight", trainable=is_train)
            self.fc1_meta_cmd_bias = _bias_variable(shape=[self.meta_cmd_dim], name="fc1_meta_cmd_bias", trainable=is_train)
            self.fc2_meta_cmd_weight = _fc_weight_variable(shape=[self.meta_cmd_dim, meta_cmd_flatten_size], name="fc2_meta_cmd_weight", trainable=is_train)
            self.fc2_meta_cmd_bias = _bias_variable(shape=[meta_cmd_flatten_size], name="fc2_meta_cmd_bias", trainable=is_train)

            meta_cmd_E_flatten_size = int(np.prod(self.meta_cmd_E_size))
            self.fc1_meta_cmd_E_weight = _fc_weight_variable(shape=[concat_dim, meta_cmd_E_flatten_size], name="fc1_meta_cmd_E_weight", trainable=is_train)
            self.fc1_meta_cmd_E_bias = _bias_variable(shape=[meta_cmd_E_flatten_size], name="fc1_meta_cmd_E_bias", trainable=is_train)

    def infer(self, hero, monster, turret, minion, stat, top_k):
        with tf.variable_scope('CEN'):
            ### unit feature ###
            # For intuitive display, we first set the number of units of each type to 1.
            # Units of the same type share parameters and will be merged by max pooling.
            unit_result_list = []
            fc1_hero_result = tf.nn.relu((tf.matmul(hero, self.fc1_hero_weight) + self.fc1_hero_bias),
                                         name="fc1_hero_result")
            fc2_hero_result = tf.nn.relu((tf.matmul(fc1_hero_result, self.fc2_hero_weight) + self.fc2_hero_bias),
                                         name="fc2_hero_result")
            fc3_hero_result = tf.add(tf.matmul(fc2_hero_result, self.fc3_hero_weight), self.fc3_hero_bias,
                                     name="fc3_hero_result")
            pool_hero_result = maxpooling([fc3_hero_result], 1, self.hero_dim // 4, name='hero_units')
            unit_result_list.append(pool_hero_result)

            fc1_monster_result = tf.nn.relu((tf.matmul(monster, self.fc1_monster_weight) + self.fc1_monster_bias),
                                         name="fc1_monster_result")
            fc2_monster_result = tf.nn.relu((tf.matmul(fc1_monster_result, self.fc2_monster_weight) + self.fc2_monster_bias),
                                         name="fc2_monster_result")
            fc3_monster_result = tf.add(tf.matmul(fc2_monster_result, self.fc3_monster_weight), self.fc3_monster_bias,
                                     name="fc3_monster_result")
            pool_monster_result = maxpooling([fc3_monster_result], 1, self.unit_dim // 4, name='monster_units')
            unit_result_list.append(pool_monster_result)

            fc1_turret_result = tf.nn.relu((tf.matmul(turret, self.fc1_turret_weight) + self.fc1_turret_bias),
                                         name="fc1_turret_result")
            fc2_turret_result = tf.nn.relu((tf.matmul(fc1_turret_result, self.fc2_turret_weight) + self.fc2_turret_bias),
                                         name="fc2_turret_result")
            fc3_turret_result = tf.add(tf.matmul(fc2_turret_result, self.fc3_turret_weight), self.fc3_turret_bias,
                                     name="fc3_turret_result")
            pool_turret_result = maxpooling([fc3_turret_result], 1, self.unit_dim // 4, name='turret_units')
            unit_result_list.append(pool_turret_result)
            
            fc1_minion_result = tf.nn.relu((tf.matmul(minion, self.fc1_minion_weight) + self.fc1_minion_bias),
                                         name="fc1_minion_result")
            fc2_minion_result = tf.nn.relu((tf.matmul(fc1_minion_result, self.fc2_minion_weight) + self.fc2_minion_bias),
                                         name="fc2_minion_result")
            fc3_minion_result = tf.add(tf.matmul(fc2_minion_result, self.fc3_minion_weight), self.fc3_minion_bias,
                                     name="fc3_minion_result")
            pool_minion_result = maxpooling([fc3_minion_result], 1, self.unit_dim // 4, name='minion_units')
            unit_result_list.append(pool_minion_result)
            concat_units = tf.concat(unit_result_list, axis=-1, name='concat_units')

            ### in-game stats feature ###
            fc1_stat_result = tf.nn.relu((tf.matmul(stat, self.fc1_stat_weight) + self.fc1_stat_bias),
                                         name="fc1_stat_result")
            fc2_stat_result = tf.add(tf.matmul(fc1_stat_result, self.fc2_stat_weight), self.fc2_stat_bias,
                                         name="fc2_stat_result")

            ### meta-command predict ###
            concat_all = tf.concat([concat_units, fc2_stat_result], axis=1, name="concat_all_fc")
            fc1_meta_cmd_result = tf.nn.relu((tf.matmul(concat_all, self.fc1_meta_cmd_weight) + self.fc1_meta_cmd_bias),
                                         name="fc1_meta_cmd_result")
            fc2_meta_cmd_result = tf.add(tf.matmul(fc1_meta_cmd_result, self.fc2_meta_cmd_weight), self.fc2_meta_cmd_bias,
                                     name="fc2_meta_cmd_result")

            softmax_prob = tf.nn.softmax(fc2_meta_cmd_result)

            topk_softmax_prob = topk_softmax(fc2_meta_cmd_result, top_k)
            meta_cmd = tf.stop_gradient(sample_from_prob(topk_softmax_prob))

            ### meta-command E auxiliary training ###
            fc1_meta_cmd_E_result = tf.add(tf.matmul(concat_all, self.fc1_meta_cmd_E_weight), self.fc1_meta_cmd_E_bias,
                                             name="fc1_meta_cmd_E_result")
            softmax_E_prob = tf.nn.softmax(fc1_meta_cmd_E_result)

            return softmax_prob, softmax_E_prob, meta_cmd