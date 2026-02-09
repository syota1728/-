"""
ACO: Number of Ants vs Learning Speed
(epoch and average distance at learning time)
"""

import random
import matplotlib.pyplot as plt

# =====================
# 共通パラメータ
# =====================
ILIMIT = 150      # 繰り返し回数
Q = 3
RHO = 0.8
STEP = 10
EPSILON = 0.15
SEED = 32767

THRESHOLD = 0.22  # 「大きく減少」と判定する割合

# コスト（固定）
cost = [
    [1]*STEP,
    [5]*STEP
]

# =====================
# ACO関数群
# =====================
def walk(cost, pheromone, mstep, NOA):
    for m in range(NOA):
        for s in range(STEP):
            if (random.random() < EPSILON or
                abs(pheromone[0][s] - pheromone[1][s]) < 0.1):
                mstep[m][s] = random.randint(0, 1)
            else:
                mstep[m][s] = 0 if pheromone[0][s] > pheromone[1][s] else 1


def update(cost, pheromone, mstep, NOA):
    sum_lm = 0.0

    # フェロモン蒸発
    for i in range(2):
        for j in range(STEP):
            pheromone[i][j] *= RHO

    # フェロモン付加
    for m in range(NOA):
        lm = 0.0
        for i in range(STEP):
            lm += cost[mstep[m][i]][i]

        for i in range(STEP):
            pheromone[mstep[m][i]][i] += Q * (1.0 / (lm * lm))

        sum_lm += lm

    return sum_lm / NOA


def run_aco(NOA):
    random.seed(SEED)

    average_walk = [0 for _ in range(ILIMIT)]
    pheromone = [[0.0 for _ in range(STEP)] for _ in range(2)]
    mstep = [[0 for _ in range(STEP)] for _ in range(NOA)]

    for i in range(ILIMIT):
        walk(cost, pheromone, mstep, NOA)
        average_walk[i] = update(cost, pheromone, mstep, NOA)

    return average_walk


def detect_learning_epoch(avg_walk, threshold):
    """
    最初に大きく改善した epoch と
    そのときの平均距離を返す
    """
    for i in range(1, len(avg_walk)):
        if avg_walk[i] < avg_walk[i - 1] * (1 - threshold):
            return i + 1, avg_walk[i]  # epoch(1始まり), 距離
    return None, None


# =====================
# 実験本体
# =====================
NOA_list = list(range(10, 101, 10))
learning_epochs = []
learning_distances = []

print("NOA, learning epoch, average distance")

for noa in NOA_list:
    avg_walk = run_aco(noa)
    epoch, distance = detect_learning_epoch(avg_walk, THRESHOLD)

    learning_epochs.append(epoch)
    learning_distances.append(distance)

    print(f"{noa}, {epoch}, {distance}")

# =====================
# 可視化（2軸）
# =====================
fig, ax1 = plt.subplots()

ax1.plot(NOA_list, learning_epochs, marker='o')
ax1.set_xlabel("Number of Ants (NOA)")
ax1.set_ylabel("Epoch of First Significant Improvement")
ax1.grid(axis='y')

ax2 = ax1.twinx()
ax2.plot(NOA_list, learning_distances, marker='s', color="tab:orange")
ax2.set_ylabel("Average Distance at Learning Epoch")

plt.title("Learning Speed and Quality vs Number of Ants (ACO)")
plt.tight_layout()
plt.show()
