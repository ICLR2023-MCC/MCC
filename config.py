class Config:
    player_num = 5
    value_head = 5
    batch_size = 10
    lstm_step = 16

class CEN_config:
    unit_dim = 256
    stat_dim = 256
    meta_cmd_dim = 256

    softmax_k = 20
    init_learning_rate = 0.0001
    epoch = 2000000

class CS_config:
    unit_dim = 256
    stat_dim = 256
    invisible_dim = 256
    cmd_query_dim = 64
    cmd_key_dim = 64
    fused_dim = 256

    init_learning_rate = 0.0002

class MCCAN_config:
    unit_dim = 256
    stat_dim = 256
    invisible_dim = 256
    action_query_dim = 128
    action_key_dim = 128

    lstm_unit_size = 1024
    # What, How (Move X), How (Move Y), Skill (Move X), Skill (Move Y), Who (Target Unit)
    action_size_list = [14, 9, 9, 9, 9, 5]

    # extrinsic rewards, intrinsic reward
    value_head_list = [Config.value_head, 1]

    init_learning_rate = 0.0002