import os
import glob
import matplotlib.pyplot as plt


def read_rewards(file_path):
    """
    读取Rewards.txt文件中的数据，每行一个数字，返回一个列表。
    """
    rewards = []
    try:
        with open(file_path, 'r') as file:
            for line_num, line in enumerate(file, start=1):
                stripped_line = line.strip()
                if stripped_line:  # 确保行不为空
                    try:
                        reward = float(stripped_line)
                        rewards.append(reward)
                    except ValueError:
                        print(f"警告：无法将第 {line_num} 行的内容转换为浮点数：'{stripped_line}'")
    except FileNotFoundError:
        print(f"错误：文件未找到：{file_path}")
    except Exception as e:
        print(f"读取文件时发生错误：{e}")

    return rewards


def plot_rewards_single(file_path, rewards, output_dir=None):
    """
    为单个Rewards文件绘制折线图。如果提供了output_dir，则将图像保存到该目录。
    """
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, marker='o', linestyle='-', color='b')
    plt.title(f'Rewards Over Episodes - {os.path.basename(file_path)}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)

    if output_dir:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        # 构建保存路径
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        save_path = os.path.join(output_dir, f"{base_name}_plot.png")
        plt.savefig(save_path)
        print(f"图像已保存到：{save_path}")
    else:
        plt.show()
    plt.close()


def main():
    # 目录路径
    directory = "/home/a325/tang/energy1011/GRL_TrainedModels/DQN2024_12_04-20:16"

    # 获取所有 .txt 文件
    file_pattern = os.path.join(directory, "*.txt")
    txt_files = glob.glob(file_pattern)

    if not txt_files:
        print(f"在目录中未找到任何 .txt 文件：{directory}")
        return

    # 输出目录（可选）
    output_dir = os.path.join(directory, "Plots")

    # 遍历所有 .txt 文件并绘制
    for file_path in txt_files:
        rewards = read_rewards(file_path)
        if rewards:
            plot_rewards_single(file_path, rewards, output_dir=output_dir)
        else:
            print(f"文件中没有可绘制的数据：{file_path}")

    print("所有图像已完成绘制。")


if __name__ == "__main__":
    main()
