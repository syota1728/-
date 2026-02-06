"""
aco.pyプログラム
蟻コロニー最適化法(aco）の例題プログラム
acoにより最適値を学習します
使い方 C:\>python aco.py
"""

# モジュールのインポート
import math
import random

# グローバル変数
NOA = 20 #蟻の個体数
ILIMIT = 50 #繰り返しの回数
Q = 3 #フェロモン更新の定数
RHO = 0.8 #蒸発の定数
STEP = 10 #道のりのステップ数
EPSILON = 0.15 #行動選択のランダム性の決定
SEED = 32767 #乱数のシード

# 下請け関数の定義
# update()関数
def update(cost, pheromone, mstep):
	"""フェロモンの更新"""
	sum_lm = 0.0 #蟻の歩いた合計距離
	#フェロモンの蒸発
	for i in range(2):
		for j in range(STEP):
			pheromone[i][j] *= RHO
	
	#蟻による上塗り
	for m in range(NOA):
		#個体mの移動距離1mの計算
		lm = 0.0
		for i in range(STEP):
			lm += cost[mstep[m][i]][i]
		#フェロモンの上塗り
		for i in range(STEP):
			pheromone[mstep[m][i]][i] += Q * (1.0/(lm * lm))
		sum_lm += lm
	#蟻の歩いた平均距離を出力
	print(sum_lm/NOA)
	return
#update()関数の終わり

#walk()関数
def walk(cost, pheromone, mstep):
	"""蟻を歩かせる"""
	for m in range(NOA):
		for s in range(STEP):
			#ε-greedy法による行動選択
			if( (random.random() < EPSILON) or (abs(pheromone[0][s] - pheromone[1][s])<0.1) ):
				#ランダムに行動
				mstep[m][s] = random.randint(0,1)
			else:
				#フェロモン濃度により選択
				if pheromone[0][s] > pheromone[1][s]:
					mstep[m][s] = 0
				else:
					mstep[m][s] = 1
	#蟻の挙動の出力
	print(mstep)
	return
#walk()関数の終わり

#メイン実行部
cost = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],[5 ,5 ,5, 5, 5, 5, 5, 5, 5, 5]] #各ステップのコスト
pheromone = [[0.0 for i in range(STEP)] for j in range(2)] #各ステップのフェロモン
mstep = [[0 for i in range(STEP)] for j in range(NOA)] #蟻が歩いた過程

#乱数の初期化
random.seed(SEED)

#最適化の本体
for i in range(ILIMIT):
	#フェロモンの状態出力
	print(i)
	print(pheromone)
	#蟻を歩かせる
	walk(cost, pheromone, mstep)
	#フェロモンの更新
	update(cost, pheromo
	
	ne, mstep)
#フェロモンの状態出力
print(i)
print(pheromone)

#aco.pyの終わり
