import os
import sys
import optparse
import traci
import numpy as np
from sumolib import checkBinary
from Env_DQN.DQN import *
import datetime
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from config import config

# -------------------------------
# 配置：全局参数定义
# -------------------------------
Training = True
num_hv = config.num_hv
num_av = config.num_av
num_lane = config.num_lane
n_episodes = config.n_episodes
# 定义warmup步长
Warmup_Steps = 1000

Testing = False
test_episodes = 1000
load_dir = 'GRL_TrainedModels/DQN2022_09_18-20_23'
test_dir = 'GRL_TrainedModels/DQN2022_09_12-16:48/test'


# -------------------------------
# 工具函数模块
# -------------------------------
def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options


# -------------------------------
# 环境状态模块
# 处理车辆的状态信息并准备强化学习模型所需的输入数据
# -------------------------------
def get_step(id_info, information):
    N = information['n_vehicles']
    num_hv = information['n_hv']  # maximum number of HDVs
    num_lanes = information['n_lanes']
    ids = id_info['ids']
    AVs = id_info['AVs']
    HVs = id_info['HVs']
    Pos_x = id_info['Pos_x']
    Pos_y = id_info['Pos_y']
    Vel = id_info['Vel']
    LaneIndex = id_info['LaneIndex']
    Edge = id_info['Edge']

    # 初始化状态空间矩阵，邻接矩阵和mask
    states = np.zeros([N, 6 + num_lanes])
    adjacency = np.zeros([N, N])
    mask = np.zeros(N)
    intention_dic = {'HV': 0, 'AV': 1}
    light_dic = {'yrrGGGGrrrrryrrGGGGrrrrr': 0, 'grrrrrGGGrrrgrrrrrGGGrrr': 1, 'GrrrrrgrrGGGGrrrrrgrrGGG': 2,
                 'GGGrrrgrrrrrGGGrrrgrrrrr': 3,
                 'GrryyyGrrrrrGrryyyGrrrrr': 4, 'GrrrrrGyyrrrGrrrrrGyyrrr': 5, 'GrrrrrGrryyyGrrrrrGrryyy': 6,
                 'GyyrrrGrrrrrGyyrrrGrrrrr': 7}

    if AVs:
        # numerical data (speed, location)
        speeds = np.array(Vel).reshape(-1, 1)  # 将速度转化为numpy格式的列向量
        xs = np.array(Pos_x).reshape(-1, 1)
        ys = np.array(Pos_y).reshape(-1, 1)
        road = np.array(Edge).reshape(-1, 1)
        # categorical data 1 hot encoding: (lane location, intention)
        lanes_column = np.array(LaneIndex)  # 当前环境中的车辆所在车道的编号
        lanes = np.zeros([len(ids), num_lanes])  # 初始化车道onehot矩阵（当前车辆数量x车道数量）
        lanes[np.arange(len(ids)), lanes_column] = 1  # 根据每辆车当前所处的车道，在矩阵对应位置赋值为1
        types_column = np.array([intention_dic[traci.vehicle.getTypeID(i)] for i in ids])  # 将自动驾驶车辆和人类驾驶车辆分别标记为0，1
        intention = np.zeros([len(ids), 2])  # 初始化intention矩阵（当前车辆数量x车辆种类）
        intention[np.arange(len(ids)), types_column] = 1  # 根据自动驾驶车辆和人类驾驶车辆类型，在矩阵对应位置赋值为1
        # types_column_l = np.array([light_dic[traci.trafficlight.getRedYellowGreenState('J3')] for i in ids]) # 不存在红绿灯，所以不需要
        # lights = np.zeros([len(ids), 8])  # 初始化light矩阵（当前车辆数量x车辆种类）
        # lights[np.arange(len(ids)), types_column_l] = 1
        # observed_states = np.c_[xs, ys, speeds, lanes, road,intention,lights[:,:4]]  # 将上述相关矩阵按列合成为状态观测矩阵
        observed_states = np.c_[xs, ys, speeds, lanes, road, intention]  # 去掉了红绿灯

        # assemble into the NxF states matrix
        # 将上述对环境的观测储存至状态矩阵中
        # print("len(HVs):", len(HVs))
        # print("states.shape:", states.shape)
        # print("observed_states.shape:", observed_states.shape)
        states[:len(HVs), :] = observed_states[:len(HVs), :]
        states[num_hv:num_hv + len(AVs), :] = observed_states[len(HVs):, :]

        # 生成邻接矩阵
        # 使用sklearn库中的欧几里德距离函数计算环境中两两车辆的水平距离（x坐标，维度当前车辆x当前车辆）
        dist_matrix_x = euclidean_distances(xs)
        dist_matrix_y = euclidean_distances(ys)
        dist_matrix = np.sqrt(dist_matrix_x * dist_matrix_x + dist_matrix_y * dist_matrix_y)
        adjacency_small = np.zeros_like(dist_matrix)  # 根据dist_matrix生成维度相同的全零邻接矩阵
        adjacency_small[dist_matrix < 20] = 1
        adjacency_small[-len(AVs):, -len(AVs):] = 1  # 将RL车辆之间在邻接矩阵中进行赋值

        # assemble into the NxN adjacency matrix (这部分程序存疑)
        # 将上述small邻接矩阵储存至稠密邻接矩阵中
        adjacency[:len(HVs), :len(HVs)] = adjacency_small[:len(HVs), :len(HVs)]
        adjacency[num_hv:num_hv + len(AVs), :len(HVs)] = adjacency_small[len(HVs):, :len(HVs)]
        adjacency[:len(HVs), num_hv:num_hv + len(AVs)] = adjacency_small[:len(HVs), len(HVs):]
        adjacency[num_hv:num_hv + len(AVs), num_hv:num_hv + len(AVs)] = adjacency_small[len(HVs):,
                                                                        len(HVs):]

        # 构造mask矩阵
        mask[num_hv:num_hv + len(AVs)] = np.ones(len(AVs))

    return states, adjacency, mask


# 将sumo的车辆状态传递进来
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
    Pos_y = []
    Vel = []
    LaneIndex = []
    Edge = []
    drastic_veh = []
    NO = 0
    for ID in ids:
        Pos_x.append(traci.vehicle.getPosition(ID)[0])
        Pos_y.append(traci.vehicle.getPosition(ID)[1])
        Vel.append(traci.vehicle.getSpeed(ID))
        current_lane = traci.vehicle.getLaneIndex(ID)
        if isinstance(rl_actions, torch.Tensor):
            rl_actions = rl_actions.cpu().numpy()
        # print(rl_actions)
        rl_actions2 = rl_actions.copy()
        rl_actions3 = rl_actions.copy()
        rl_actions2[rl_actions2 > 2] = 1

        # print(next_lane)
        rl_actions3[rl_actions3 < 5] = 14

        LaneIndex.append(current_lane)
        Edge.append(EdgeList.index(traci.vehicle.getRoadID(ID)))
        if traci.vehicle.getTypeID(ID) == 'AV':
            AVs.append(ID)

            # print(traci.vehicle.getSpeed(ID) + (rl_actions3[NO] - 18) / 2 * 0.1)

        elif traci.vehicle.getTypeID(ID) == 'HV':
            HVs.append(ID)
        rl_actions2 = rl_actions2[num_hv:num_hv + len(AVs)]
        rl_actions3 = rl_actions3[num_hv:num_hv + len(AVs)]  # 只对自动驾驶车辆产生
        if len(AVs) != 0:
            next_lane = np.clip(current_lane + rl_actions2[NO] - 1, 0, 2)
            traci.vehicle.setLaneChangeMode(ID, 0b000000000100)
            traci.vehicle.setSpeedMode(ID, 0b011111)
            traci.vehicle.changeLane(ID, next_lane, 100)  # 100代表变道时间是100ms
            traci.vehicle.setSpeed(ID, traci.vehicle.getSpeed(ID) + (rl_actions3[NO] - 14) / 2)
            # print(rl_actions3-8)
            NO += 1
        # if isinstance(rl_actions, torch.Tensor):
        #     rl_actions = rl_actions.cpu().numpy()
        #     # print(rl_actions)
        # rl_actions = rl_actions - 1
        # rl_actions2 = rl_actions.copy()
        # rl_actions3 = rl_actions2.copy()
        # rl_actions4 = rl_actions2.copy()
        # rl_actions3[rl_actions3 > 1] = 0  # langchange action
        #
        # rl_actions4[rl_actions4 < 2] = 18  # acc action
        #
        # # rl_actions3 = rl_actions3[num_hv:num_hv + len(AVs)]
        # # rl_actions4 = rl_actions4[num_hv:num_hv + len(AVs)]
        # next_lane = np.clip(current_lane + rl_actions3[NO], 0, 2)
        # # print(next_lane)
        #
        #
        #
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
               'Pos_y': Pos_y, 'Vel': Vel, 'LaneIndex': LaneIndex, 'Edge': Edge, 'drastic_veh': drastic_veh,
               'drastic_veh_id': drastic_veh_id}

    return id_list


# -------------------------------
# 奖励函数模块
# -------------------------------
def get_reward(info_of_state, information):
    """
    计算多智能体的奖励。

    参数：
        info_of_state (dict): 包含当前状态信息的字典。
        information (dict): 包含环境配置信息的字典。

    返回：
        float: 总奖励。
    """
    # 配置权重
    w_speed = 1.0
    w_distance = 1.0
    w_collision = -100.0
    w_wait = -0.1
    w_energy = -0.05
    w_smooth = -0.1
    w_lane_change = -0.2

    # 初始化奖励
    reward = 0.0

    # 获取所有车辆的ID
    ids = traci.vehicle.getIDList()
    AVs = info_of_state['AVs']
    HVs = info_of_state['HVs']

    # 期望速度
    desired_speed_av = information['Max_speed_AV']
    desired_speed_hv = information['Max_speed_HV']

    # 处理AVs
    for av_id in AVs:
        # 获取当前速度
        speed = traci.vehicle.getSpeed(av_id)

        # 1. 速度奖励
        speed_error = abs(speed - desired_speed_av)
        if speed_error < 2.0:
            R_speed = 1.0  # 接近期望速度
        elif speed > desired_speed_av:
            R_speed = -0.5  # 速度过快
        else:
            R_speed = -0.5  # 速度过慢
        reward += w_speed * R_speed

        # 2. 安全距离奖励
        leader = traci.vehicle.getLeader(av_id)
        follower = traci.vehicle.getFollower(av_id)
        R_distance = 0.0

        # 前车距离
        if leader:
            leader_id, gap = leader
            if gap < 5.0:
                R_distance -= 1.0  # 距离过近
            elif gap > 20.0:
                R_distance -= 0.5  # 距离过远
            else:
                R_distance += 1.0  # 距离合适
        else:
            R_distance += 1.0  # 没有前车，视为安全

        # 后车距离
        if follower:
            follower_id, gap = follower
            if gap < 5.0:
                R_distance -= 1.0  # 距离过近
            elif gap > 20.0:
                R_distance -= 0.5  # 距离过远
            else:
                R_distance += 1.0  # 距离合适
        else:
            R_distance += 1.0  # 没有后车，视为安全

        reward += w_distance * R_distance

        # 3. 平稳驾驶奖励
        acc = traci.vehicle.getAcceleration(av_id)
        if abs(acc) < 1.0:
            R_smooth = 1.0  # 平稳驾驶
        else:
            R_smooth = -0.5  # 急加速或急刹车
        reward += w_smooth * R_smooth

        # 4. 变道惩罚
        lane_change = traci.vehicle.getLaneChangeState(av_id, -1)
        # Lane change states: 0 - not changing, 1 - preparing, 2 - lane changing, etc.
        if lane_change != 0:
            R_lane_change = -1.0  # 进行变道
        else:
            R_lane_change = 0.0  # 不变道
        reward += w_lane_change * R_lane_change

        # 5. 能源消耗惩罚
        energy_consumption = traci.vehicle.getElectricityConsumption(av_id)
        R_energy = energy_consumption  # 假设能量消耗与直接惩罚成正比
        reward += w_energy * R_energy

        # 6. 碰撞惩罚
        collision_vehicles = traci.simulation.getCollidingVehiclesIDList()
        if av_id in collision_vehicles:
            R_collision = 1.0  # 碰撞发生
            reward += w_collision * R_collision

    # 通常，HVs的行为不受RL控制，可以不给予奖励或设计额外的逻辑

    # 7. 等待时间惩罚
    for av_id in AVs:
        waiting_time = traci.vehicle.getWaitingTime(av_id)
        R_wait = waiting_time  # 等待时间越长，惩罚越大
        reward += w_wait * R_wait

    return reward


# 定义warmup步长记录
warmup_count = 0
now_time = datetime.datetime.now()
now_time = datetime.datetime.strftime(now_time, '%Y_%m_%d-%H:%M')
save_dir = 'GRL_TrainedModels/DQN' + now_time
try:
    os.makedirs(save_dir)
except:
    pass

info_dict = {'n_vehicles': num_hv + num_av, 'n_hv': num_hv, 'n_av': num_av, 'n_lanes': num_lane,
             'Max_speed_AV': 50, 'Max_speed_HV': 30}

# -------------------------------
# 训练与测试模块
# -------------------------------
if Training:
    Rewards = []  # 初始化奖励矩阵以进行数据保存
    Loss = []  # 初始化Loss矩阵以进行数据保存
    Episode_Steps = []  # 初始化步长矩阵保存每一episode的任务完成时的步长
    Average_Q = []  # 初始化平均Q值矩阵保存每一episode的平均Q值
    collision = []
    energy = []
    AVspeed = []
    AVchangelane = []

    # 创建多智能体各自的网络结构
    GRL_Nets, agent = Create_DQN(num_hv, num_av, info_dict)

    print("#------------------------------------#")
    print("#----------Training Begins-----------#")
    print("#------------------------------------#")

    for i in range(1, n_episodes + 1):

        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")

        if __name__ == "__main__":
            options = get_options()
            if options.nogui:
                sumoBinary = checkBinary('sumo')
            else:
                sumoBinary = checkBinary('sumo-gui')
            traci.start(['sumo', "-c", "Env_Net/roundabout.sumocfg"])
            # 初始化actions动作
            num_agents = num_av + num_hv
            actions = np.zeros(num_agents)  # 生成original动作
            list_info = check_state(actions)
            obs = get_step(list_info, info_dict)
            R = 0  # 行为奖励
            t = 0  # 时间步长
            A_e = 0
            A_s1 = []
            A_s = 0
            A_change = 0
            crash_name = 0
            # warmup_count = 0
            num_actions = config.num_actions
            for step in range(0, 4000):
                # ------动作生成------ #
                if warmup_count <= Warmup_Steps:  # 进行warmup
                    actions = np.random.choice(
                        np.arange(num_actions), num_agents)  # 生成随机动作
                else:
                    for i in range(num_agents):  # agent与环境进行交互
                        if i < num_hv:
                            actions[i] = 0
                        else:
                            action = agent.choose_action(obs, i)
                            actions[i] = action[i]

                list_info = check_state(actions)
                obs_next = get_step(list_info, info_dict)
                reward = get_reward(list_info, info_dict)
                exist_AV = []
                ids = traci.vehicle.getIDList()
                AV_energy = 0
                rl_speed = 0
                for id in ids:
                    if traci.vehicle.getTypeID(id) == 'AV':
                        AV_energy += traci.vehicle.getElectricityConsumption(id)
                        exist_AV.append(id)
                        rl_speed += traci.vehicle.getSpeed(id)
                if len(exist_AV) != 0:
                    AV_speed = rl_speed / len(exist_AV)
                    A_s1.append(AV_speed)
                A_change += len(list_info['drastic_veh_id'])
                R += reward
                t += 1
                crash_num1 = traci.simulation.getCollidingVehiclesNumber()
                crash_name += crash_num1
                A_e += AV_energy

                warmup_count += 1
                done = (len(exist_AV) == 0) and (step > 1000)
                # ------将交互结果储存到经验回放池中------ #
                agent.store_transition(obs, actions, reward, obs_next, done)
                # ------进行策略更新------ #
                agent.learn()
                # ------环境观测更新------ #
                obs = obs_next
                traci.simulationStep(step*0.1)
                print("当前AV：", exist_AV)
                if done:
                    break
            traci.close()
            # ------记录训练数据------ #
            # 获取训练数据
            training_data = agent.get_statistics()
            loss = training_data[0]
            q = training_data[1]
            # 记录训练数据
            for i in A_s1:
                A_s += i
            A_s = A_s / len(A_s1)
            Rewards.append(R)  # 记录Rewards
            Episode_Steps.append(t)  # 记录Steps
            Loss.append(loss)  # 记录loss
            Average_Q.append(q)  # 记录平均Q值
            collision.append(crash_name)
            energy.append(A_e)
            AVspeed.append(A_s)
            AVchangelane.append(A_change)

            if i % 1 == 0:
                print('Training Episode:', i, 'Reward:', R, 'Loss:', loss, 'Average_Q:', q)
            plt.figure(1)
            plt.subplot(2, 4, 1)
            plt.title('Rewards')
            plt.plot(Rewards)
            plt.subplot(2, 4, 2)
            plt.title('Episode_Steps')
            plt.plot(Episode_Steps)
            plt.subplot(2, 4, 3)
            plt.title('Loss')
            plt.plot(Loss)
            plt.subplot(2, 4, 4)
            plt.title('Average_Q')
            plt.plot(Average_Q)
            plt.subplot(2, 4, 5)
            plt.title('collision')
            plt.plot(collision)
            plt.subplot(2, 4, 6)
            plt.title('energy')
            plt.plot(energy)
            plt.subplot(2, 4, 7)
            plt.title('AVspeed')
            plt.plot(AVspeed)
            plt.subplot(2, 4, 8)
            plt.title('AVchangelane')
            plt.plot(AVchangelane)
            plt.show(block=False)
            plt.pause(1)
            print("one episode finished")
    print('Training Finished.')
    # 模型保存
    agent.save_model(save_dir)
    # 保存训练过程中的各项数据
    np.save(save_dir + "/Rewards", Rewards)
    np.save(save_dir + "/Episode_Steps", Episode_Steps)
    np.save(save_dir + "/Loss", Loss)
    np.save(save_dir + "/Average_Q", Average_Q)
    np.save(save_dir + "/collision", collision)
    np.save(save_dir + "/energy", energy)
    np.save(save_dir + "/AVspeed", AVspeed)
    np.save(save_dir + "/AVchangelane", AVchangelane)
    np.savetxt(save_dir + '/rewards.txt', Rewards, delimiter=',')
    np.savetxt(save_dir + '/Episode_Steps.txt', Episode_Steps, delimiter=',')
    np.savetxt(save_dir + '/losses.txt', Loss, delimiter=',')
    np.savetxt(save_dir + '/AverageQ.txt', Average_Q, delimiter=',')
    np.savetxt(save_dir + '/collision.txt', collision, delimiter=',')
    np.savetxt(save_dir + '/energy.txt', energy, delimiter=',')
    np.savetxt(save_dir + '/AVspeed.txt', AVspeed, delimiter=',')
    np.savetxt(save_dir + '/AVchangelane.txt', AVchangelane, delimiter=',')
    plt.figure(1)
    plt.savefig(save_dir + '/datas.png')

if Testing:
    Rewards = []  # 初始化奖励矩阵以进行数据保存
    GRL_Net, GRL_model = Create_DQN(num_hv, num_av, info_dict)
    GRL_model.load_model(load_dir)

    print("#-------------------------------------#")
    print("#-----------Testing Begins------------#")
    print("#-------------------------------------#")
    for i in range(1, test_episodes + 1):
        traci.start(['sumo-gui', "-c", "Env_Net/well.sumocfg"])
        action = np.zeros(GRL_Net.num_agents)  # 生成original动作
        list_info = check_state(action)
        obs = get_step(list_info, info_dict)
        R = 0  # 行为奖励
        t = 0  # 时间步长
        for step in range(0, 2500):
            # ------动作生成------ #
            # 生成随机动作

            action = GRL_model.choose_action(obs)  # agent与环境进行交互

            list_info = check_state(action)
            obs_next = get_step(list_info, info_dict)
            reward = get_reward(list_info, info_dict)
            R += reward
            t += 1
            warmup_count += 1
            done = False
            # ------环境观测更新------ #
            obs = obs_next
            traci.simulationStep()
            if done:
                break
        traci.close()
        # traci.start(['sumo', "-c", "Env_Net/well.sumocfg"])
        # action = np.zeros(GRL_Net.num_agents)  # 生成original动作
        # list_info = check_state(action)
        # obs = get_step(list_info, info_dict)
        # if 'SUMO_HOME' in os.environ:
        #     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        #     sys.path.append(tools)
        # else:
        #     sys.exit("please declare environment variable 'SUMO_HOME'")
        #
        # R = 0
        # t = 0
        #
        # action = GRL_model.test_action(obs)
        # list_info = check_state(action)
        # obs_next = get_step(list_info, info_dict)
        # reward = get_reward(list_info, info_dict)
        # R += reward
        # t += 1
        # done = False
        # if done :
        #     break
        # traci.close()

        print('Evaluation Episode:', i, 'Reward:', R)

    print('Evaluation Finished')

    # 测试数据保存
    np.savetxt(test_dir + "/Test", Rewards)