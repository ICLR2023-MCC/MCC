from tf_env import tf
from utils import _fc_weight_variable, _bias_variable, _conv_weight_variable, maxpooling
import numpy as np
from config import Config, MCCAN_config

class MCCAN:
    def __init__(self, spatial_size, hero_size, monster_size, turret_size, minion_size, stat_size, invisible_size, meta_cmd_size):
        self.lstm_time_steps = Config.lstm_step
        self.lstm_unit_size = MCCAN_config.lstm_unit_size
        self.batch_size = Config.batch_size

        self.spatial_size = spatial_size
        self.hero_size = hero_size
        self.monster_size = monster_size
        self.turret_size = turret_size
        self.minion_size = minion_size
        self.stat_size = stat_size
        self.invisible_size = invisible_size
        self.meta_cmd_size = meta_cmd_size

        self.player_num = Config.player_num
        self.hero_dim = MCCAN_config.hero_dim
        self.unit_dim = MCCAN_config.unit_dim
        self.stat_dim = MCCAN_config.stat_dim
        self.invisible_dim = MCCAN_config.invisible_dim
        self.action_query_dim = MCCAN_config.action_query_dim
        self.action_key_dim = MCCAN_config.action_key_dim

        self.action_size_list = MCCAN_config.action_size_list
        self.value_head = int(sum(MCCAN_config.value_head_list))

    def build(self, is_train):
        with tf.variable_scope('MCCAN'):
            self.conv1_kernel = _conv_weight_variable(shape=[5, 5, self.spatial_size[0], 18], name="spatial_conv1_kernel")
            self.conv1_bias = _bias_variable(shape=[18], name="spatial_conv1_bias")
            self.conv2_kernel = _conv_weight_variable(shape=[3, 3, 18, 12], name="spatial_conv2_kernel")
            self.conv2_bias = _bias_variable(shape=[12], name="spatial_conv2_bias")
            spatial_flatten_size = 768

            hero_flatten_size = int(np.prod(self.hero_size))
            self.fc1_hero_weight = _fc_weight_variable(shape=[hero_flatten_size, self.hero_dim], name="fc1_hero_weight", trainable=is_train)
            self.fc1_hero_bias = _bias_variable(shape=[self.hero_dim], name="fc1_hero_bias", trainable=is_train)
            self.fc2_hero_weight = _fc_weight_variable(shape=[self.hero_dim, self.hero_dim // 2], name="fc2_hero_weight", trainable=is_train)
            self.fc2_hero_bias = _bias_variable(shape=[self.hero_dim // 2], name="fc2_hero_bias", trainable=is_train)
            self.fc3_hero_weight = _fc_weight_variable(shape=[self.hero_dim // 2, self.action_query_dim], name="fc3_hero_weight", trainable=is_train)
            self.fc3_hero_bias = _bias_variable(shape=[self.action_query_dim], name="fc3_hero_bias", trainable=is_train)

            monster_flatten_size = int(np.prod(self.monster_size))
            self.fc1_monster_weight = _fc_weight_variable(shape=[monster_flatten_size, self.unit_dim], name="fc1_monster_weight", trainable=is_train)
            self.fc1_monster_bias = _bias_variable(shape=[self.unit_dim], name="fc1_monster_bias", trainable=is_train)
            self.fc2_monster_weight = _fc_weight_variable(shape=[self.unit_dim, self.unit_dim // 2], name="fc2_monster_weight", trainable=is_train)
            self.fc2_monster_bias = _bias_variable(shape=[self.unit_dim // 2], name="fc2_monster_bias", trainable=is_train)
            self.fc3_monster_weight = _fc_weight_variable(shape=[self.unit_dim // 2, self.action_query_dim], name="fc3_monster_weight", trainable=is_train)
            self.fc3_monster_bias = _bias_variable(shape=[self.action_query_dim], name="fc3_monster_bias", trainable=is_train)

            turret_flatten_size = int(np.prod(self.turret_size))
            self.fc1_turret_weight = _fc_weight_variable(shape=[turret_flatten_size, self.unit_dim], name="fc1_turret_weight", trainable=is_train)
            self.fc1_turret_bias = _bias_variable(shape=[self.unit_dim], name="fc1_turret_bias", trainable=is_train)
            self.fc2_turret_weight = _fc_weight_variable(shape=[self.unit_dim, self.unit_dim // 2], name="fc2_turret_weight", trainable=is_train)
            self.fc2_turret_bias = _bias_variable(shape=[self.unit_dim // 2], name="fc2_turret_bias", trainable=is_train)
            self.fc3_turret_weight = _fc_weight_variable(shape=[self.unit_dim // 2, self.action_query_dim], name="fc3_turret_weight", trainable=is_train)
            self.fc3_turret_bias = _bias_variable(shape=[self.action_query_dim], name="fc3_turret_bias", trainable=is_train)

            minion_flatten_size = int(np.prod(self.minion_size))
            self.fc1_minion_weight = _fc_weight_variable(shape=[minion_flatten_size, self.unit_dim], name="fc1_minion_weight", trainable=is_train)
            self.fc1_minion_bias = _bias_variable(shape=[self.unit_dim], name="fc1_minion_bias", trainable=is_train)
            self.fc2_minion_weight = _fc_weight_variable(shape=[self.unit_dim, self.unit_dim // 2], name="fc2_minion_weight", trainable=is_train)
            self.fc2_minion_bias = _bias_variable(shape=[self.unit_dim // 2], name="fc2_minion_bias", trainable=is_train)
            self.fc3_minion_weight = _fc_weight_variable(shape=[self.unit_dim // 2, self.action_query_dim], name="fc3_minion_weight", trainable=is_train)
            self.fc3_minion_bias = _bias_variable(shape=[self.action_query_dim], name="fc3_minion_bias", trainable=is_train)

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

            meta_cmd_flatten_size = int(np.prod(self.meta_cmd_size))
            concat_all_size = spatial_flatten_size + self.action_query_dim * 4 + self.stat_dim // 4 + meta_cmd_flatten_size
            self.fc_concat_weight = _fc_weight_variable(shape=[concat_all_size, self.lstm_unit_size], name="fc_concat_weight", trainable=is_train)
            self.fc_concat_bias = _bias_variable(shape=[self.lstm_unit_size], name="fc_concat_bias", trainable=is_train)

            self.lstm = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.lstm_unit_size, forget_bias=1.0, trainable=is_train)

            # label
            self.fc_action_weight_list = []
            self.fc_action_bias_list = []
            for label_index in range(len(self.action_size_list)):
                if (label_index == (len(self.action_size_list) - 1)):
                    fc_label_weight = _fc_weight_variable(shape=[self.lstm_unit_size, self.action_key_dim], name="fc_label_%d_weight" % (label_index))
                    fc_label_bias = _bias_variable(shape=[self.action_key_dim], name="fc_label_%d_bias" % (label_index))
                else:
                    fc_label_weight = _fc_weight_variable(shape=[self.lstm_unit_size, self.action_size_list[label_index]], name="fc_label_%d_weight" % (label_index))
                    fc_label_bias = _bias_variable(shape=[self.action_size_list[label_index]], name="fc_label_%d_bias" % (label_index))
                self.fc_action_weight_list.append(fc_label_weight)
                self.fc_action_bias_list.append(fc_label_bias)

            self.fc_target_attention_weight = _fc_weight_variable(shape=[self.action_query_dim, self.action_query_dim], name="fc_target_attention_weight")

            # value
            concat_state_szie = self.lstm_unit_size + self.invisible_dim // 4
            self.fc_value_weight = _fc_weight_variable(shape=[concat_state_szie, self.value_head], name="fc_value_weight", trainable=is_train)
            self.fc_value_bias = _bias_variable(shape=[self.value_head], name="fc_value_bias", trainable=is_train)

    def infer(self, player_feature_list, select_cmd_list):
        with tf.variable_scope('MCCAN'):
            public_embed_list = []
            private_embed_list = []
            invisible_embed_list = []
            target_embed_list = []
            for player_index in range(self.player_num):
                one_target_embed_list = []
                feature_list = ['spatial', 'hero', 'monster', 'turret', 'minion', 'stat', 'invisible']
                spatial, hero, monster, turret, minion, stat, invisible = [player_feature_list[player_index][k] for k in feature_list]
                select_cmd = select_cmd_list[player_index]

                transpose_spatial = tf.transpose(spatial, perm=[0, 2, 3, 1], name='transpose_spatial_%d' % player_index)
                conv1_result = tf.nn.relu((tf.nn.conv2d(transpose_spatial, self.conv1_kernel, strides=[1, 1, 1, 1], padding="SAME") + self.conv1_bias), name="conv1_result_%d" % player_index)
                pool_conv1_result = tf.nn.max_pool(conv1_result, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID', name="pool_conv1_result_%d" % player_index)
                temp_conv2_result = tf.nn.bias_add(tf.nn.conv2d(pool_conv1_result, self.conv2_kernel, strides=[1, 1, 1, 1], padding="SAME"), self.conv2_bias, name="temp_conv2_result_%d" % player_index)
                conv2_result = tf.transpose(temp_conv2_result, perm=[0, 3, 1, 2], name="conv2_result_%d" % player_index)
                conv2_dim = int(np.prod(conv2_result.get_shape()[1:]))
                flatten_conv2_result = tf.reshape(conv2_result, shape=[-1, conv2_dim], name="flatten_conv2_result_%d" % player_index)

                ### unit feature ###
                # For intuitive display, we first set the number of units of each type to 1.
                # Units of the same type share parameters and will be merged by max pooling.
                fc1_hero_result = tf.nn.relu((tf.matmul(hero, self.fc1_hero_weight) + self.fc1_hero_bias), name="fc1_hero_result_%d" % player_index)
                fc2_hero_result = tf.nn.relu((tf.matmul(fc1_hero_result, self.fc2_hero_weight) + self.fc2_hero_bias), name="fc2_hero_result_%d" % player_index)
                fc3_hero_result = tf.add(tf.matmul(fc2_hero_result, self.fc3_hero_weight), self.fc3_hero_bias, name="fc3_hero_result_%d" % player_index)
                pool_hero_result = maxpooling([fc3_hero_result], 1, self.action_query_dim, name="hero_units_%d" % player_index)
                one_target_embed_list.append(pool_hero_result)

                fc1_monster_result = tf.nn.relu((tf.matmul(monster, self.fc1_monster_weight) + self.fc1_monster_bias), name="fc1_monster_result_%d" % player_index)
                fc2_monster_result = tf.nn.relu((tf.matmul(fc1_monster_result, self.fc2_monster_weight) + self.fc2_monster_bias), name="fc2_monster_result_%d" % player_index)
                fc3_monster_result = tf.add(tf.matmul(fc2_monster_result, self.fc3_monster_weight), self.fc3_monster_bias, name="fc3_monster_result_%d" % player_index)
                pool_monster_result = maxpooling([fc3_monster_result], 1, self.action_query_dim, name="monster_units_%d" % player_index)
                one_target_embed_list.append(pool_monster_result)

                fc1_turret_result = tf.nn.relu((tf.matmul(turret, self.fc1_turret_weight) + self.fc1_turret_bias), name="fc1_turret_result_%d" % player_index)
                fc2_turret_result = tf.nn.relu((tf.matmul(fc1_turret_result, self.fc2_turret_weight) + self.fc2_turret_bias), name="fc2_turret_result_%d" % player_index)
                fc3_turret_result = tf.add(tf.matmul(fc2_turret_result, self.fc3_turret_weight), self.fc3_turret_bias, name="fc3_turret_result_%d" % player_index)
                pool_turret_result = maxpooling([fc3_turret_result], 1, self.action_query_dim, name="turret_units_%d" % player_index)
                one_target_embed_list.append(pool_turret_result)

                fc1_minion_result = tf.nn.relu((tf.matmul(minion, self.fc1_minion_weight) + self.fc1_minion_bias), name="fc1_minion_result_%d" % player_index)
                fc2_minion_result = tf.nn.relu((tf.matmul(fc1_minion_result, self.fc2_minion_weight) + self.fc2_minion_bias), name="fc2_minion_result_%d" % player_index)
                fc3_minion_result = tf.add(tf.matmul(fc2_minion_result, self.fc3_minion_weight), self.fc3_minion_bias, name="fc3_minion_result_%d" % player_index)
                pool_minion_result = maxpooling([fc3_minion_result], 1, self.action_query_dim, name="minion_units_%d" % player_index)
                one_target_embed_list.append(pool_minion_result)

                ### in-game stats feature ###
                fc1_stat_result = tf.nn.relu((tf.matmul(stat, self.fc1_stat_weight) + self.fc1_stat_bias), name="fc1_stat_result_%d" % player_index)
                fc2_stat_result = tf.add(tf.matmul(fc1_stat_result, self.fc2_stat_weight), self.fc2_stat_bias, name="fc2_stat_result_%d" % player_index)

                ### invisible invisible information (only for training) ###
                fc1_invisible_result = tf.nn.relu((tf.matmul(invisible, self.fc1_invisible_weight) + self.fc1_invisible_bias), name="fc1_invisible_result_%d" % player_index)
                fc2_invisible_result = tf.nn.relu((tf.matmul(fc1_invisible_result, self.fc2_invisible_weight) + self.fc2_invisible_bias), name="fc2_invisible_result_%d" % player_index)
                fc3_invisible_result = tf.add(tf.matmul(fc2_invisible_result, self.fc3_invisible_weight), self.fc3_invisible_bias, name="fc3_invisible_result_%d" % player_index)
                invisible_embed_list.append(fc3_invisible_result)

                # add none target embedding
                none_use_target_embed = tf.ones_like(one_target_embed_list[0], dtype=tf.float32, name='none_use_target_embed') * 0.1
                one_target_embed_list.insert(0, none_use_target_embed)
                target_embed_list.append(one_target_embed_list)
                concat_all = tf.concat([flatten_conv2_result, pool_hero_result, pool_monster_result, pool_turret_result, pool_minion_result, fc2_stat_result, select_cmd], axis=1, name="concat_all")
                fc_concat_result = tf.nn.relu((tf.matmul(concat_all, self.fc_concat_weight) + self.fc_concat_bias), name="fc_concat_result_%d" % player_index)

                public_size = self.lstm_unit_size // 4
                one_public_embed, one_private_embed = tf.split(fc_concat_result, [public_size, self.lstm_unit_size - public_size], axis=1)
                public_embed_list.append(one_public_embed)
                private_embed_list.append(one_private_embed)

            # share information
            player_public_concat_result = tf.concat(public_embed_list, axis=1, name="public_concat_result")
            reshape_player_public = tf.reshape(player_public_concat_result, shape=[-1, len(public_embed_list), self.lstm_unit_size // 4, 1], name="reshape_hero_public")
            pool_player_public = tf.nn.max_pool(reshape_player_public, [1, len(public_embed_list), 1, 1], [1, 1, 1, 1], padding='VALID', name="pool_public")
            player_public_input_dim = int(np.prod(pool_player_public.get_shape()[1:]))
            reshape_pool_hero_public = tf.reshape(pool_player_public, shape=[-1, player_public_input_dim], name="reshape_pool_public")

            # lstm
            lstm_embed_list = []
            for player_index in range(self.player_num):
                player_embed_result_list = [reshape_pool_hero_public, private_embed_list[player_index]]
                player_embed_result = tf.concat(player_embed_result_list, axis=1, name="embed_concat_result_%d" % player_index)
                reshape_embed_result = tf.reshape(player_embed_result, [-1, self.lstm_time_steps, self.lstm_unit_size], name="reshape_embed_concat_result")

                lstm_initial_state = self.lstm.zero_state(self.batch_size, dtype=tf.float32)
                lstm_outputs, lstm_last_states = tf.nn.dynamic_rnn(self.lstm, reshape_embed_result, initial_state=lstm_initial_state)
                reshape_lstm_outputs_result = tf.reshape(lstm_outputs, [-1, self.lstm_unit_size], name="reshape_lstm_outputs_result")
                lstm_embed_list.append(reshape_lstm_outputs_result)

            player_action_list = []
            player_value_list = []
            # predict action and value
            for player_index in range(self.player_num):
                one_action_list = []
                with tf.variable_scope("player%d_action" % player_index):
                    # action output
                    for action_index in range(len(self.action_size_list)):
                        # target attention for Target Unit
                        if (action_index == (len(self.action_size_list) - 1)):
                            fc_action_key = tf.add(tf.matmul(lstm_embed_list[player_index], self.fc_action_weight_list[action_index]), self.fc_action_bias_list[action_index], name="player%d_fc_action_key" % (player_index))
                            stack_action_query = tf.reshape(tf.stack(target_embed_list[player_index], axis=1), shape=[-1, self.action_query_dim], name="stack_action_query")
                            fc_action_query = tf.matmul(stack_action_query, self.fc_target_attention_weight)
                            reshape_fc_action_query = tf.reshape(fc_action_query, shape=[-1, len(target_embed_list[player_index]), self.action_query_dim], name="reshape_fc_action_query")
                            reshape_fc_action_key = tf.reshape(fc_action_key, shape=[-1, self.action_key_dim, 1], name="reshape_fc_label_attention")
                            temp_action_result = tf.matmul(reshape_fc_action_query, reshape_fc_action_key)
                            action_logits = tf.reshape(temp_action_result, shape=[-1, int(np.prod(temp_action_result.get_shape()[1:]))], name="hero%d_fc_label_%d_result" % (player_index, action_index))
                        else:
                            action_logits = tf.add(tf.matmul(lstm_embed_list[player_index], self.fc_action_weight_list[action_index]), self.fc_action_bias_list[action_index], name="player%d_fc_action_%d_result" % (player_index, action_index))
                        softmax_prob = tf.nn.softmax(action_logits)
                        one_action_list.append(softmax_prob)
                    player_action_list.append(one_action_list)

                with tf.variable_scope("player%d_value" % player_index):
                    # value output
                    concat_state = tf.concat([lstm_embed_list[player_index], invisible_embed_list[player_index]], axis=1, name="concat_state")
                    fc_value_result = tf.add(tf.matmul(concat_state, self.fc_value_weight), self.fc_value_bias, name="player%d_fc_value_result" % (player_index))
                    player_value_list.append(fc_value_result)

        return player_action_list, player_value_list