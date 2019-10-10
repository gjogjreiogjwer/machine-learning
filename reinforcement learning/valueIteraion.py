# 价值迭代

import gym
import numpy as np

env = gym.make('FrozenLake-v0')
env.reset()
# print (env.P)
def value_interation(env, gamma = 0.8, no_of_iterations = 2000):
	value_table = np.zeros(env.observation_space.n)
	threshold = 1e-20
	for i in range(no_of_iterations):
		updated_value_table = np.copy(value_table)
		for state in range(env.observation_space.n):
			q_value = []
			for action in range(env.action_space.n):
				next_states_rewards = []
				for next_sr in env.P[state][action]:
					# next_sr = (0.3333333333333333, 8,       0.0           , False)
					# next_sr = (状态转移概率,        下一个状态,得到reward的概率,游戏是否结束)
					trans_prob, next_state, reward_prob, _ = next_sr
					# 下一状态t的动作状态价值 = 转移到t状态的概率 × （ env反馈的reward + γ × t状态的当前价值 ）
					next_states_rewards.append((trans_prob*(reward_prob+gamma*updated_value_table[next_state])))
				# 将某一动作执行后，所有可能的t+1状态的价值加起来，就是在t状态采取a动作的价值 
				q_value.append(np.sum(next_states_rewards))
			value_table[state] = max(q_value)
		if np.sum(np.fabs(updated_value_table - value_table)) <= threshold:
			print ('value interation converged at iteration %d' % (i+1))
			break
	return value_table


def extract_policy(value_table, gamma = 1.0):
	policy = np.zeros(env.observation_space.n)
	for state in range(env.observation_space.n):
		# 将价值迭代的过程再走一遍，但是不再更新value function，而是选出每个状态下对应最大价值的动作
		q_table = np.zeros(env.action_space.n)
		for action in range(env.action_space.n):
			for next_sr in env.P[state][action]:
				trans_prob, next_state, reward_prob, _ = next_sr
				q_table[action] += (trans_prob*(reward_prob+gamma*value_table[next_state]))
		policy[state] = np.argmax(q_table)
	return policy



a = value_interation(env)
b = extract_policy(a)
print (b)




















