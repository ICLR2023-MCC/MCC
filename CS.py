from tf_env import tf
from utils import _fc_weight_variable, _bias_variable, maxpooling, _conv_weight_variable
import numpy as np
from config import CS_config

class CS:
    def __init__(self, hero_size, monster_size, turret_size, minion_size, stat_size, invisible_size, meta_cmd_size, value_head):
        self.hero_size = hero_size
        self.monster_size = monster_size
        self.turret_size = turret_size
        self.minion_size = minion_size
        self.stat_size = stat_size
        self.invisible_size = invisible_size
        self.meta_cmd_size = meta_cmd_size

        self.hero_dim = CS_config.hero_dim
        self.unit_dim = CS_config.unit_dim
        self.stat_dim = CS_config.stat_dim
        self.invisible_dim = CS_config.invisible_dim
        self.cmd_query_dim = CS_config.cmd_query_dim
        self.cmd_key_dim = CS_config.cmd_key_dim
        self.fused_dim = CS_config.fused_dim

        self.value_head = value_head

    def build(self, is_train):
        with tf.variable_scope('CS'):
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

            invisible_flatten_size = int(np.prod(self.invisible_size))
            self.fc1_invisible_weight = _fc_weight_variable(shape=[invisible_flatten_size, self.invisible_dim], name="fc1_invisible_weight", trainable=is_train)
            self.fc1_invisible_bias = _bias_variable(shape=[self.invisible_dim], name="fc1_invisible_bias", trainable=is_train)
            self.fc2_invisible_weight = _fc_weight_variable(shape=[self.invisible_dim, self.invisible_dim // 2], name="fc2_invisible_weight", trainable=is_train)
            self.fc2_invisible_bias = _bias_variable(shape=[self.invisible_dim // 2], name="fc2_invisible_bias", trainable=is_train)
            self.fc3_invisible_weight = _fc_weight_variable(shape=[self.invisible_dim // 2, self.invisible_dim // 4], name="fc3_invisible_weight", trainable=is_train)
            self.fc3_invisible_bias = _bias_variable(shape=[self.invisible_dim // 4], name="fc3_invisible_bias", trainable=is_train)

            self.cmd_conv1_kernel = _conv_weight_variable(shape=[2, 2, 1, 5], name="cmd_conv1_kernel", trainable=is_train)
            self.cmd_conv1_bias = _bias_variable(shape=[5], name="cmd_conv1_bias", trainable=is_train)
            self.cmd_conv2_kernel = _conv_weight_variable(shape=[3, 3, 5, 8], name="cmd_conv2_kernel", trainable=is_train)
            self.cmd_conv2_bias = _bias_variable(shape=[8], name="cmd_conv2_bias", trainable=is_train)
            self.conv_output_size = 128

            concat_state_dim = self.hero_dim // 4 + self.unit_dim // 4 * 3 + self.stat_dim // 4
            self.fc_state_gate_weight = _fc_weight_variable(shape=[self.conv_output_size, concat_state_dim], name="fc_inter_gmlp_gate_weight", trainable=is_train)
            self.fc_state_gate_bias = _bias_variable(shape=[concat_state_dim], name="fc_inter_gmlp_gate_bias", trainable=is_train)
            self.fc_cmd_gate_weight = _fc_weight_variable(shape=[concat_state_dim, self.conv_output_size], name="fc_inter_gmlp_cmd_gate_weight", trainable=is_train)
            self.fc_cmd_gate_bias = _bias_variable(shape=[self.conv_output_size], name="fc_inter_gmlp_cmd_gate_bias", trainable=is_train)

            concat_fused_dim = self.conv_output_size + concat_state_dim
            self.fc1_fused_weight = _fc_weight_variable(shape=[concat_fused_dim, self.fused_dim], name="fc1_fused_weight", trainable=is_train)
            self.fc1_fused_bias = _bias_variable(shape=[self.fused_dim], name="fc1_fused_bias", trainable=is_train)
            self.fc2_fused_weight = _fc_weight_variable(shape=[self.fused_dim, self.fused_dim // 2], name="fc2_fused_weight", trainable=is_train)
            self.fc2_fused_bias = _bias_variable(shape=[self.fused_dim // 2], name="fc2_fused_bias", trainable=is_train)

            self.cmd_query_weight = _fc_weight_variable(shape=[self.conv_output_size, self.cmd_query_dim], name="fc_cmd_query_weight", trainable=is_train)
            self.cmd_query_bias = _bias_variable(shape=[self.cmd_query_dim], name="fc_cmd_query_bias", trainable=is_train)
            self.fused_key_weight = _fc_weight_variable(shape=[self.fused_dim // 2, self.cmd_key_dim], name="fc_fused_key_weight", trainable=is_train)
            self.fused_key_bias = _bias_variable(shape=[self.cmd_key_dim], name="fc_xfused_key_bias", trainable=is_train)

            concat_train_dim = self.fused_dim // 2 + self.invisible_dim // 4
            self.fc_value_weight = _fc_weight_variable(shape=[concat_train_dim, self.value_head], name="fc_value_weight", trainable=is_train)
            self.fc_value_bias = _bias_variable(shape=[self.value_head], name="fc_value_bias", trainable=is_train)

            self.fc_attention_weight = _fc_weight_variable(shape=[self.cmd_query_dim, self.cmd_query_dim], name="fc_attention_weight", trainable=is_train)

    def infer(self, hero, monster, turret, minion, stat, invisible, cmd_set):
        with tf.variable_scope('CS'):
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

            ### invisible invisible information (only for training) ###
            fc1_invisible_result = tf.nn.relu((tf.matmul(invisible, self.fc1_invisible_weight) + self.fc1_invisible_bias),
                                         name="fc1_invisible_result")
            fc2_invisible_result = tf.nn.relu((tf.matmul(fc1_invisible_result, self.fc2_invisible_weight) + self.fc2_invisible_bias),
                                         name="fc2_invisible_result")
            fc3_invisible_result = tf.add(tf.matmul(fc2_invisible_result, self.fc3_invisible_weight), self.fc3_invisible_bias,
                                        name="fc3_invisible_result")

            ### meta-command set feature ###
            cmd_map_result_list = []
            cmd_query_list = []
            for index in range(len(cmd_set)):
                meta_cmd = tf.reshape(cmd_set[index], [-1] + self.meta_cmd_size)
                transpose_cmd_map = tf.transpose(meta_cmd, perm=[0, 2, 3, 1], name="transpose_cmd_%d_map" % index)
                with tf.variable_scope("cmd_%d_conv1" % index):
                    cmd_conv1_result = tf.nn.relu((tf.nn.conv2d(transpose_cmd_map, self.cmd_conv1_kernel, strides=[1, 2, 2, 1], padding="VALID") + self.cmd_conv1_bias), name="cmd_conv1_result")
                with tf.variable_scope("cmd_%d_conv2" % index):
                    temp_cmd_conv2_result = tf.nn.bias_add(tf.nn.conv2d(cmd_conv1_result, self.cmd_conv2_kernel, strides=[1, 1, 1, 1], padding="VALID"), self.cmd_conv2_bias, name="temp_cmd_conv2_result")
                    cmd_conv2_result = tf.transpose(temp_cmd_conv2_result, perm=[0, 3, 1, 2], name="cmd_conv2_result")
                with tf.variable_scope("cmd_%d_flatten_conv2" % index):
                    conv2_dim = int(np.prod(cmd_conv2_result.get_shape()[1:]))
                    cmd_map_result = tf.reshape(cmd_conv2_result, shape=[-1,conv2_dim], name="cmd_map_result")
                cmd_map_result_list.append(cmd_map_result)

                fc_cmd_query_result = tf.add(tf.matmul(cmd_map_result, self.cmd_query_weight), self.cmd_query_bias, name="fc_cmd_info_result_%d" % (index))
                cmd_query_list.append(fc_cmd_query_result)

            ### gating mechanism ###
            concat_state = tf.concat([concat_units, fc2_stat_result], axis=1, name="concat_state")
            pool_cmd_map_set = maxpooling(cmd_map_result_list, len(cmd_map_result_list), self.conv_output_size, name='cmd_set')
            with tf.variable_scope("cmd_inter_gmlp"):
                # state as gate
                cmd_gate = tf.add(tf.matmul(concat_state, self.fc_cmd_gate_weight), self.fc_cmd_gate_bias, name="fc_cmd_gate_result")
                cmd_gmlp_inter = pool_cmd_map_set * cmd_gate
                # cmd as gate
                state_gate = tf.add(tf.matmul(pool_cmd_map_set, self.fc_state_gate_weight), self.fc_state_gate_bias, name="fc_gate_result")
                state_gmlp_inter = concat_state * state_gate
                # fusion
                state_gmlp_inter = state_gmlp_inter + concat_state
                cmd_gmlp_inter = cmd_gmlp_inter + pool_cmd_map_set
                concat_fused = tf.concat([state_gmlp_inter, cmd_gmlp_inter], axis=1, name="concat_fused")

            fc1_fused_result = tf.nn.relu((tf.matmul(concat_fused, self.fc1_fused_weight) + self.fc1_fused_bias), name="fc1_fused_result")
            fc2_fused_result = tf.nn.elu((tf.matmul(fc1_fused_result, self.fc2_fused_weight) + self.fc2_fused_bias), name="fc2_fused_result")
            fc_fused_key = tf.add(tf.matmul(cmd_map_result, self.cmd_query_weight), self.cmd_query_bias, name="fc_fused_key")

            ### V (state value) predict ###
            concat_train_info = tf.concat([fc2_fused_result, fc3_invisible_result], axis=1, name="concat_train_info")
            value = tf.add(tf.matmul(concat_train_info, self.fc_value_weight), self.fc_value_bias, name="fc_value_result")

            ### target attention ###
            reshape_fc_fused_key = tf.reshape(fc_fused_key, shape=[-1, self.cmd_key_dim, 1], name="reshape_fc_fused_key")
            # key-value
            stack_cmd_query = tf.stack(cmd_query_list, axis=1)
            reshape_cmd_query = tf.reshape(stack_cmd_query, shape=[-1, self.cmd_query_dim], name="reshape_cmd_query")
            fc_cmd_query = tf.matmul(reshape_cmd_query, self.fc_attention_weight)
            reshape_fc_cmd_query = tf.reshape(fc_cmd_query, shape=[-1, len(cmd_query_list), self.cmd_query_dim], name="reshape_fc_cmd_query")

            ### Q (state-action value) predict ###
            temp_q_value = tf.matmul(reshape_fc_cmd_query, reshape_fc_fused_key)
            q_value = tf.reshape(temp_q_value, shape=[-1, len(cmd_query_list)], name="reshape_q_value_result")

            ### select meta-command ###
            select_index = tf.argmax(q_value, axis=-1)
            selext_mat = tf.expand_dims(tf.one_hot(select_index, len(cmd_query_list)), 1, name='selext_mat')
            concat_cmd_map = tf.concat(tf.stack(cmd_set, axis=1), axis=1, name='concat_cmd_map')

            select_cmd = tf.stop_gradient(tf.squeeze(tf.matmul(selext_mat, concat_cmd_map, name='select_cmd'), axis=[1]))

            return value, q_value, select_cmd