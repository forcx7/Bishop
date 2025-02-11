import os
import sys
import optparse
import traci
import math
import numpy as np

from GRL_Library.agent.AVDQN_agent import dtype
from sumolib import checkBinary
from Env_DQN.DQN import *
import datetime
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from config import config

Training = True
num_hv = config.num_hv
num_av = config.num_av
num_lane = config.num_lane
n_episodes = config.n_episodes
# 定义warmup步长
Warmup_Steps = 500

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
                         default=True, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options


# -------------------------------
# 状态空间
# -------------------------------
def get_step(id_info, information, i):
    N = information['n_vehicles']
    num_hv = information['n_hv']  # maximum number of HDVs
    num_av = information['n_av']
    num_lanes = information['n_lanes']
    # ids = id_info['ids']
    ids = traci.vehicle.getIDList()
    AVs = id_info['AVs']
    HVs = id_info['HVs']
    Pos_x = id_info['Pos_x']
    Pos_av_x = id_info['Pos_av_x']
    Pos_y = id_info['Pos_y']
    Pos_av_y = id_info['Pos_av_y']
    Vel = id_info['Vel']
    LaneIndex = id_info['LaneIndex']
    Edge = id_info['Edge']
    av_ids = [veh_id for veh_id in ids if traci.vehicle.getTypeID(veh_id) == 'AV']

    # 初始化状态空间矩阵，邻接矩阵和mask
    # states_small = []
    adjacencies_small = []
    state = np.zeros([20, 6])
    adjacency = np.zeros([20, 20])
    # intention_dic = {'HV': 0, 'AV': 1}

    if f't_{i}' in av_ids:  # 如果出现AV

        # 分布式节点特征矩阵
        # near_small = []
        # near = []
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
        max_nearby = 19  # 大于20就截断，小于20不管
        state = [[veh[0], veh[1], speed, lanes[0], lanes[1], 1]]
        for other_id in ids:
            if other_id == av_id:
                continue
            pos = traci.vehicle.getPosition(other_id)
            distance = math.hypot(veh[0] - pos[0], veh[1] - pos[1])  # 距离
            if distance < 20 and nearby < max_nearby:
                speed = traci.vehicle.getSpeed(other_id)
                # lane = traci.vehicle.getLaneIndex(other_id)
                # edge = traci.vehicle.getRoadID(other_id)
                lanes_column = np.array(traci.vehicle.getLaneIndex(other_id))  # 当前环境中的车辆所在车道的编号
                lanes = np.zeros([num_lanes])  # 初始化车道onehot矩阵（1x车道数量）
                lanes[lanes_column] = 1  # 根据所处的车道，在矩阵对应位置赋值为1
                state.append([pos[0], pos[1], speed, lanes[0], lanes[1], 0])
                nearby += 1
        state = np.array(state)
        # near_small.append(nearby)
        # states_small.append(state)
        # near = np.array(near_small)
        # print("near: ",near)
        # print("states_small: ", states_small)
        # states = np.array(states_small, dtype=object) # 希望 states 是一个对象数组，而不是尝试将它们转换为标准的矩阵或数组？？？
        # print("states: ", states)



        # 生成邻接矩阵
        # 把states中的state拿出来，提取出x、y的信息（前两列）
        # print("state: ", state)
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
        # adjacencies_small.append(adjacency)  # 不论有无AV，格式必须固定，需要倒腾矩阵
        # adjacencies = np.array(adjacencies_small, dtype=object)

        # 构造mask矩阵
        # 当av存在时值为1
        # for i in range(len(near)):
        #     mask = np.ones(near[i] + 1, dtype=int)
        #     mask[1:] = int(0)
        #     masks_small.append(mask)
        # masks = np.array(masks_small, dtype=object)
    return state, adjacency


def check_state(rl_actions, i):
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

    if f't_{i}' in AVs:  # 判断 't_i' 是否是 CAV
        action_value = rl_actions  # 提取 action 数值
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
    # for ind, veh_id in enumerate(AVs):  # 这部分通过计算当前时间以及最后一次换道的时间间隔来检测车辆是否有激烈换到行为，enumerate返回索引和元素本身
    #     if rl_actions[ind] != 0 and (time_counter - traci.vehicle.getLastActionTime(veh_id) < 50):
    #         drastic_veh.append(veh_id)
    # drastic_veh_id = drastic_veh
    id_list = {'ids': ids, 'EdgeList': EdgeList, 'exist_V': exist_V, 'AVs': AVs, 'HVs': HVs, 'Pos_x': Pos_x,
               'Pos_av_x': Pos_av_x,
               'Pos_y': Pos_y, 'Pos_av_y': Pos_av_y, 'Vel': Vel, 'LaneIndex': LaneIndex, 'Edge': Edge}

    return id_list


def get_reward(info_of_state, information, i):
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
    av_id = f't_{i}'
    if av_id in AVs:
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

        # # 4. 变道惩罚
        # lane_change = traci.vehicle.getLaneChangeState(av_id, -1)
        # # Lane change states: 0 - not changing, 1 - preparing, 2 - lane changing, etc.
        # if lane_change != 0:
        #     R_lane_change = -1.0  # 进行变道
        # else:
        #     R_lane_change = 0.0  # 不变道
        # reward += w_lane_change * R_lane_change

        # # 5. 能源消耗惩罚
        # energy_consumption = traci.vehicle.getElectricityConsumption(av_id)
        # R_energy = energy_consumption  # 假设能量消耗与直接惩罚成正比
        # reward += w_energy * R_energy

        # 6. 碰撞惩罚
        # collision_vehicles = traci.simulation.getCollidingVehiclesIDList()
        # if av_id in collision_vehicles:
        #     R_collision = 1.0  # 碰撞发生
        #     reward += w_collision * R_collision
    else:
        reward = 0.0
    # 通常，HVs的行为不受RL控制，可以不给予奖励或设计额外的逻辑

    # 7. 等待时间惩罚
    # for av_id in AVs:
    #     waiting_time = traci.vehicle.getWaitingTime(av_id)
    #     R_wait = waiting_time  # 等待时间越长，惩罚越大
    #     reward += w_wait * R_wait

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

    # 创建多智能体各自的网络结构
    agent = Create_DQN(num_hv, num_av, info_dict)

    print("#------------------------------------#")
    print("#----------Training Begins-----------#")
    print("#------------------------------------#")

    for j in range(1, n_episodes + 1):
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
            traci.start([sumoBinary, "-c", "Env_Net/25115_r.sumocfg"])

            # 初始化，但感觉鸡肋没多大用
            # num_agents = num_av + num_hv
            # actions = np.zeros(num_av)  # 生成original动作
            actions = np.full(num_av,10) # 初始化不变道不变速
            list_info = check_state(actions, 0)  # 执行动作（变道换速），返回整体环境信息
            obs = get_step(list_info, info_dict, 0)  # 返回节点特征矩阵、邻接矩阵

            R = 0  # 行为奖励
            t = 0  # 时间步长
            num_actions = config.num_actions
            for step in range(0, 4000):
                # ------动作生成------ #
                for i in range(num_av):  # agent与环境进行交互
                    obs = get_step(list_info, info_dict, i)  # 返回每个agent节点特征矩阵、邻接矩阵
                    if warmup_count <= Warmup_Steps:
                        actions[i] = np.random.choice(np.arange(num_actions))  # 生成随机动作
                    else:
                        if f't_{i}' not in exist_AV:
                            # continue
                            actions[i] = 10
                        else:
                            action = agent.choose_action(obs, i)
                            actions[i] = action[0]
                    list_info = check_state(actions[i], i)  # 某个AV执行动作（变道换速），返回环境信息
                    obs_next = get_step(list_info, info_dict, i) # 某个AV的obs_next
                    reward = get_reward(list_info, info_dict, i) # 某个AV的reward
                    R += reward
                    exist_AV = []
                    ids = traci.vehicle.getIDList()
                    for id in ids:
                        if traci.vehicle.getTypeID(id) == 'AV':
                            exist_AV.append(id)
                    done = (len(exist_AV) == 0) and (step > 500)
                    # ------将交互结果储存到经验回放池中------ #
                    if f't_{i}' in exist_AV:
                        agent.store_transition(obs, actions[i], reward, obs_next, done, i)

                t += 1
                warmup_count += 1
                traci.simulationStep(step * 0.5)
                # ------进行策略更新------ #
                agent.learn(exist_AV)
                print("当前AV：", exist_AV)
                if done:
                    break
            traci.close()
            # ------记录训练数据------ #
            # 获取训练数据
            training_data = agent.get_statistics()
            loss = training_data[0]
            q = training_data[1]
            print("loss,q:", loss, q)
            # 记录训练数据
            Rewards.append(R)  # 记录Rewards
            Episode_Steps.append(t)  # 记录训练Steps
            print("Episode_Steps", Episode_Steps)
            Loss.append(loss)  # 记录loss
            Average_Q.append(q)  # 记录平均Q值
            print("Loss,Q:", Loss, Average_Q)

            if j % 1 == 0:
                print('Training Episode:', j, 'Reward:', R, 'Loss:', loss, 'Average_Q:', q)
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
    np.savetxt(save_dir + '/rewards.txt', Rewards, delimiter=',')
    np.savetxt(save_dir + '/Episode_Steps.txt', Episode_Steps, delimiter=',')
    np.savetxt(save_dir + '/losses.txt', Loss, delimiter=',')
    np.savetxt(save_dir + '/AverageQ.txt', Average_Q, delimiter=',')
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