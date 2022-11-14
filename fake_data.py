from tf_env import tf
import numpy as np

class FakeData:
    def __init__(self, batch_size, lstm_step, player_num):
        self.batch_size = batch_size * lstm_step
        self.player_num = player_num
        self.size_dict = {
            'hero': [10], # Unit feature -- Heros
            'monster': [5],  # Unit feature -- Monsters
            'turret': [5],  # Unit feature -- Turrets
            'minion': [5],  # Unit feature -- Minions
            'stat': [50], # In-game stats feature
            'spatial': [6, 17, 17], # Spatial feature
            'invisible': [50], # Invisible opponent information
            'meta_cmd': [1, 12, 12], # Meta-Command feature
            'CEN_label': [144], # CEN label -- real Meta-Command (one-hot)
            'CS_V_label': [5], # CS V label -- real r + gamma * V(S', C) for each value head
            'CS_Q_label': [1], # CS Q label -- real r_total + gamma * Q(S', C', M')
            'CS_cmd_label': [5], # CS cmd label -- real selected cmd (one-hot)
         }

    def get_size(self):
        return self.size_dict

    def get_data(self):
        players = []
        for i in range(self.player_num):
            data = {}
            for key, size in self.size_dict.items():
                if key in ['CEN_label', 'CS_cmd_label']:
                    data[key] = np.zeros(([self.batch_size] + size))
                    rand_idx = np.random.randint(0, size, self.batch_size)
                    for row, col in enumerate(rand_idx):
                        data[key][row, col] = 1
                else:
                    data[key] = np.random.rand(*([self.batch_size] + size))
            players.append(data)
        return players

    def get_placeholder(self):
        players = []
        for i in range(self.player_num):
            placeholder = {}
            for key, size in self.size_dict.items():
                placeholder[key] = tf.placeholder(tf.float32, [self.batch_size] + size, name='player_%d_%s' % (i, key))
            players.append(placeholder)
        return players

    def get_placeholder_data_dict(self, placeholder, data):
        ph_data_dict = {}
        for i in range(self.player_num):
            player_ph = placeholder[i]
            player_data = data[i]
            assert len(player_ph) == len(player_data)
            for key, size in self.size_dict.items():
                ph_data_dict[player_ph[key]] = player_data[key]
        return ph_data_dict