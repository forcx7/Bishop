# 收集所有CAV的state的俩函数
def get_step(id_info, information):
    N = information['n_vehicles']
    num_hv = information['n_hv']  # maximum number of HDVs
    num_av = information['n_av']
    num_lanes = information['n_lanes']
    ids = id_info['ids']
    AVs = id_info['AVs']
    HVs = id_info['HVs']
    Pos_x = id_info['Pos_x']
    Pos_av_x = id_info['Pos_av_x']
    Pos_y = id_info['Pos_y']
    Pos_av_y = id_info['Pos_av_y']
    Vel = id_info['Vel']
    LaneIndex = id_info['LaneIndex']
    Edge = id_info['Edge']

    # 初始化状态空间矩阵，邻接矩阵和mask
    states_small = []
    masks_small = []
    adjacencies_small = []
    states = np.zeros([1, 21, 7])
    adjacencies = np.zeros([1, 21, 21])
    masks = np.zeros([1, 1, 21])
    intention_dic = {'HV': 0, 'AV': 1}

    if AVs:  # 如果出现AV
        # # 集中式节点特征矩阵（所有车辆信息）
        # speeds = np.array(Vel).reshape(-1, 1) #将速度转化为numpy格式的列向量
        # xs = np.array(Pos_x).reshape(-1, 1)  # 将POS_x转换为numpy形式的列向量
        # ys = np.array(Pos_y).reshape(-1, 1)
        # road = np.array(Edge).reshape(-1, 1)
        # # categorical data 1 hot encoding: (lane location, intention)
        # lanes_column = np.array(LaneIndex)  # 当前环境中的车辆所在车道的编号
        # lanes = np.zeros([len(ids), num_lanes])  # 初始化车道onehot矩阵（当前车辆数量x车道数量）
        # lanes[np.arange(len(ids)), lanes_column] = 1  # 根据每辆车当前所处的车道，在矩阵对应位置赋值为1
        # types_column = np.array([intention_dic[traci.vehicle.getTypeID(i)] for i in ids]) #将自动驾驶车辆和人类驾驶车辆分别标记为0，1
        # intention = np.zeros([len(ids), 2])  # 初始化intention矩阵（当前车辆数量x车辆种类）
        # intention[np.arange(len(ids)), types_column] = 1# 根据自动驾驶车辆和人类驾驶车辆类型，在矩阵对应位置赋值为1
        # types_column_l = np.array([light_dic[traci.trafficlight.getRedYellowGreenState('J3')] for i in ids]) # 不存在红绿灯，所以不需要
        # observed_states = np.c_[xs, ys, speeds, lanes, road, intention]  # 去掉了红绿灯
        # # 将上述对环境的观测储存至状态矩阵中
        # print("len(HVs):", len(HVs))
        # states[:len(HVs), :] = observed_states[:len(HVs), :]
        # states[num_hv:num_hv + len(AVs), :] = observed_states[len(HVs):, :]

        # 分布式节点特征矩阵
        av_ids = [veh_id for veh_id in traci.vehicle.getIDList() if traci.vehicle.getTypeID(veh_id) == 'AV']
        near_small = []
        near = []
        for i in range(num_av):
            if f't_{i}' in av_ids:
                av_id = f't_{i}'
                veh = traci.vehicle.getPosition(av_id)
                speed = traci.vehicle.getSpeed(av_id)
                # lane = traci.vehicle.getLaneIndex(av_id)
                lanes_column = np.array(traci.vehicle.getLaneIndex(av_id))  # 当前环境中的车辆所在车道的编号
                lanes = np.zeros([num_lanes])  # 初始化车道onehot矩阵（1x车道数量）
                lanes[lanes_column] = 1  # 根据每辆车当前所处的车道，在矩阵对应位置赋值为1
                # edge = traci.vehicle.getRoadID(av_id)
                # Find nearby vehicles
                nearby = 0
                max_nearby = 20  # 大于20就截断，小于20不管
                state = [[veh[0], veh[1], speed, lanes[0], lanes[1], lanes[2], 1]]
                for other_id in ids:
                    if other_id == av_id:
                        continue
                    pos = traci.vehicle.getPosition(other_id)
                    distance = math.hypot(veh[0] - pos[0], veh[1] - pos[1])  # 距离
                    if distance < 20 and nearby <= max_nearby:
                        speed = traci.vehicle.getSpeed(other_id)
                        # lane = traci.vehicle.getLaneIndex(other_id)
                        # edge = traci.vehicle.getRoadID(other_id)
                        lanes_column = np.array(traci.vehicle.getLaneIndex(other_id))  # 当前环境中的车辆所在车道的编号
                        lanes = np.zeros([num_lanes])  # 初始化车道onehot矩阵（1x车道数量）
                        lanes[lanes_column] = 1  # 根据所处的车道，在矩阵对应位置赋值为1
                        state.append([pos[0], pos[1], speed, lanes[0], lanes[1], lanes[2], 0])
                        nearby += 1
                state = np.array(state)
            else:
                state = np.zeros([21, 7])
            # near_small.append(nearby)
            states_small.append(state)
        # near = np.array(near_small)
        # print("near: ",near)
        # print("states_small: ", states_small)
        states = np.array(states_small, dtype=object) # 希望 states 是一个对象数组，而不是尝试将它们转换为标准的矩阵或数组？？？
        # print("states: ", states)



        # 生成邻接矩阵
        # 把states中的state拿出来，提取出x、y的信息
        for state in states:
            # 提取 x 和 y 坐标（前两列）
            # print("state: ", state)
            state = np.array(state)
            coords = state[:, :2]
            x_coords = coords[:, 0].reshape(-1, 1)  # x 坐标列向量
            y_coords = coords[:, 1].reshape(-1, 1)  # y 坐标列向量
            dist_matrix_x = euclidean_distances(x_coords)
            dist_matrix_y = euclidean_distances(y_coords)
            dist_matrix = np.sqrt(dist_matrix_x * dist_matrix_x + dist_matrix_y * dist_matrix_y)  # 每对AV车辆间的距离
            adjacency = np.zeros_like(dist_matrix)  # 根据dist_matrix生成维度相同的全零邻接矩阵
            mask = dist_matrix < 20  # 创建一个布尔掩码，标记distmatrix中小于20的元素
            normalized_values = np.round(dist_matrix[mask] / 20, 2)
            adjacency[mask] = normalized_values
            # adjacency[dist_matrix < 20] = int(1) # 需要更改？？？
            adjacencies_small.append(adjacency)  # 不论有无AV，格式必须固定，需要倒腾矩阵
        adjacencies = np.array(adjacencies_small, dtype=object)

        # # 使用sklearn库中的欧几里德距离函数计算环境中两两车辆的水平距离（x坐标，维度当前车辆x当前车辆）
        # dist_matrix_x = euclidean_distances(xs)
        # dist_matrix_y = euclidean_distances(ys)
        # dist_matrix = np.sqrt(dist_matrix_x * dist_matrix_x + dist_matrix_y * dist_matrix_y) # 每对车辆间的距离
        # adjacency_small = np.zeros_like(dist_matrix)  # 根据dist_matrix生成维度相同的全零邻接矩阵
        # adjacency_small[dist_matrix < 20] = 1
        # adjacency_small[-len(AVs):, -len(AVs):] = 1  # 将RL车辆之间在邻接矩阵中进行赋值
        # # assemble into the NxN adjacency matrix
        # # 将上述small邻接矩阵储存至稠密邻接矩阵中
        # adjacency[:len(HVs), :len(HVs)] = adjacency_small[:len(HVs), :len(HVs)]
        # adjacency[num_hv:num_hv + len(AVs), :len(HVs)] = adjacency_small[len(HVs):, :len(HVs)]
        # adjacency[:len(HVs), num_hv:num_hv + len(AVs)] = adjacency_small[:len(HVs), len(HVs):]
        # adjacency[num_hv:num_hv + len(AVs), num_hv:num_hv + len(AVs)] = adjacency_small[len(HVs):,
        #                                                                 len(HVs):]

        # 构造mask矩阵
        # 当av存在时值为1
        # for i in range(len(near)):
        #     mask = np.ones(near[i] + 1, dtype=int)
        #     mask[1:] = int(0)
        #     masks_small.append(mask)
        # masks = np.array(masks_small, dtype=object)
    return states, adjacencies
def check_state(rl_actions):
    ids = traci.vehicle.getIDList()
    EdgeList = traci.edge.getIDList()
    time_counter = traci.simulation.getTime()
    exist_V = []
    for num in range(len(traci.simulation.getArrivedIDList())):  # 获取已经到达目的地的车辆 ID 列表
        exist_V.append(traci.simulation.getArrivedIDList()[num])
    AVs = []
    HVs = []
    Pos_x = []
    Pos_av_x = []
    Pos_y = []
    Pos_av_y = []
    Vel = []
    LaneIndex = []
    Edge = []
    drastic_veh = []
    num_lane_changes = 3
    num_speed_changes = 7
    for ID in ids:
        Pos_x.append(traci.vehicle.getPosition(ID)[0])
        Pos_y.append(traci.vehicle.getPosition(ID)[1])
        if traci.vehicle.getTypeID(ID) == 'AV':
            Pos_av_x.append(traci.vehicle.getPosition(ID)[0])
            Pos_av_y.append(traci.vehicle.getPosition(ID)[1])
        Vel.append(traci.vehicle.getSpeed(ID))
        current_lane = traci.vehicle.getLaneIndex(ID)
        if isinstance(rl_actions, torch.Tensor):
            rl_actions = rl_actions.cpu().numpy()

        LaneIndex.append(current_lane)
        Edge.append(EdgeList.index(traci.vehicle.getRoadID(ID)))
        if traci.vehicle.getTypeID(ID) == 'AV':
            AVs.append(ID)
        elif traci.vehicle.getTypeID(ID) == 'HV':
            HVs.append(ID)

    for i in range(5):  # 因为 actions 数组长度为 5
        if f't_{i}' in AVs:  # 判断 't_i' 是否是 CAV
            action_value = actions[i]  # 提取对应位置的 actions 数值
            # lane_change_action = action_value // num_speed_changes  # 0,1,2；整除；左移-不变-右移
            speed_change_action = action_value % num_speed_changes  # 0-6；取余；加减速幅度
            # 加减速处理
            speed_change = 0
            speed_step = 1.5
            if speed_change_action == 0:
                speed_change = -3  # 大幅减速
            elif speed_change_action == 1:
                speed_change = -2  # 中幅减速
            elif speed_change_action == 2:
                speed_change = -1  # 小幅减速
            elif speed_change_action == 3:
                speed_change = 0  # 保持速度
            elif speed_change_action == 4:
                speed_change = 1  # 小幅加速
            elif speed_change_action == 5:
                speed_change = 2  # 中幅加速
            elif speed_change_action == 6:
                speed_change = 3  # 大幅加速
            # next_lane = np.clip(current_lane + rl_actions2[NO] - 1, 0, 2)
            # traci.vehicle.setLaneChangeMode(ID, 0b000000000100)
            # traci.vehicle.changeLane(ID, next_lane, 100)  # 100代表变道时间是100ms
            # traci.vehicle.setSpeedMode(ID, 0b011111)
            traci.vehicle.setSpeed(f't_{i}', max(0, traci.vehicle.getSpeed(f't_{i}') + speed_change * speed_step))

        # if isinstance(rl_actions, torch.Tensor):
        #     rl_actions = rl_actions.cpu().numpy()
        #     # print(rl_actions)
        # rl_actions = rl_actions - 1
        # rl_actions2 = rl_actions.copy()
        # rl_actions3 = rl_actions2.copy()
        # rl_actions4 = rl_actions2.copy()
        # rl_actions3[rl_actions3 > 1] = 0  # langchange action
        # rl_actions4[rl_actions4 < 2] = 18  # acc action
        #
        # # rl_actions3 = rl_actions3[num_hv:num_hv + len(AVs)]
        # # rl_actions4 = rl_actions4[num_hv:num_hv + len(AVs)]
        # next_lane = np.clip(current_lane + rl_actions3[NO], 0, 2)
        # # print(next_lane)

        # NO += 1
        # LaneIndex.append(current_lane)
        # Edge.append(EdgeList.index(traci.vehicle.getRoadID(ID)))
        # if traci.vehicle.getTypeID(ID) == 'AV':
        #     AVs.append(ID)
        #     traci.vehicle.changeLane(ID, next_lane, 0)
        #     traci.vehicle.setAccel(ID, traci.vehicle.getSpeed(ID) + (rl_actions4[NO] - 18) / 2 * 0.1)
        # elif traci.vehicle.getTypeID(ID) == 'HV':
        #     HVs.append(ID)
        # print(current_lane)
    for ind, veh_id in enumerate(AVs):  # 这部分通过计算当前时间以及最后一次换道的时间间隔来检测车辆是否有激烈换到行为，enumerate返回索引和元素本身
        if rl_actions[ind] != 0 and (time_counter - traci.vehicle.getLastActionTime(veh_id) < 50):
            drastic_veh.append(veh_id)
    drastic_veh_id = drastic_veh
    id_list = {'ids': ids, 'EdgeList': EdgeList, 'exist_V': exist_V, 'AVs': AVs, 'HVs': HVs, 'Pos_x': Pos_x,
               'Pos_av_x': Pos_av_x,
               'Pos_y': Pos_y, 'Pos_av_y': Pos_av_y, 'Vel': Vel, 'LaneIndex': LaneIndex, 'Edge': Edge,
               'drastic_veh': drastic_veh,
               'drastic_veh_id': drastic_veh_id}

    return id_list