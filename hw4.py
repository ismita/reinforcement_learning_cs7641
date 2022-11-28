#%%
import numpy as np
import gymnasium as gym
from gymnasium import wrappers
import time
import sys
import matplotlib.pyplot as plt
import mdptoolbox, mdptoolbox.example
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import math
import mdp


def Frozen_Lake_Experiments(frozen_lake_size):
	# 0 = left; 1 = down; 2 = right;  3 = up

	environment  = 'FrozenLake-v1'
	env = gym.make(environment,desc=generate_random_map(size=frozen_lake_size))
	env = env.unwrapped
	desc = env.unwrapped.desc


	### POLICY ITERATION ####
	time_array=[0]*10
	gamma_arr=[0]*10
	iters=[0]*10
	list_scores=[0]*10

	policy_diff_list = []
	reward_list = []

	print('POLICY ITERATION FOR FROZEN LAKE')
	for i in range(0,10):
		st=time.time()
		best_policy, k, policy_diff, score_list = policy_iteration(env, gamma = (i+0.5)/10)
		scores = evaluate_policy(env, best_policy, gamma = (i+0.5)/10)
		end=time.time()
		gamma_arr[i]=(i+0.5)/10
		list_scores[i]=np.mean(scores)
		iters[i] = k
		time_array[i]=end-st
		policy_diff_list.append(policy_diff)
		reward_list.append(score_list)


	plt.plot(gamma_arr, time_array)
	plt.xlabel('Gammas')
	plt.title(f'Frozen Lake-PI-Execution Time vs Gammas-{frozen_lake_size}*{frozen_lake_size}')
	plt.ylabel('Execution Time (s)')
	plt.grid()
	plt.show()

	plt.plot(gamma_arr,list_scores)
	plt.xlabel('Gammas')
	plt.ylabel('Average Rewards')
	plt.title(f'Frozen Lake-PI-Reward vs Gammas-{frozen_lake_size}*{frozen_lake_size}')
	plt.grid()
	plt.show()

	plt.plot(gamma_arr,iters)
	plt.xlabel('Gammas')
	plt.ylabel('Iterations to Converge')
	plt.title(f'Frozen Lake-PI-Convergence vs Gammas-{frozen_lake_size}*{frozen_lake_size}')
	plt.grid()
	plt.show()

	# plot rewards vs iterations, delta vs iterations

	for g in range(len(gamma_arr)):
		plt.plot(reward_list[g], label=f'Gamma={gamma_arr[g]}')
	# plt.plot(score_list)
	plt.legend(loc="best")
	plt.xlabel('Number of Iterations')
	plt.ylabel('Average Rewards')
	plt.title(f'Frozen Lake-PI-Reward vs Iterations-{frozen_lake_size}*{frozen_lake_size}')
	plt.grid()
	plt.show()


	for g in range(len(gamma_arr)):
		plt.plot(policy_diff_list[g], label=f'Gamma={gamma_arr[g]}')
	# plt.plot(policy_diff)
	plt.legend(loc="best")
	plt.xlabel('Number of Iterations')
	plt.ylabel('Policy Difference')
	plt.title(f'Frozen Lake-PI-Policy Difference vs Iterations-{frozen_lake_size}*{frozen_lake_size}')
	plt.grid()
	plt.show()

	### VALUE ITERATION ###
	print('VALUE ITERATION FOR FROZEN LAKE')
	time_array=[0]*10
	gamma_arr=[0]*10
	iters=[0]*10
	list_scores=[0]*10

	best_vals=[0]*10
	rewards_list = []
	value_diff_list = []
	for i in range(0,10):
		st=time.time()
		best_value,k, value_diff, score_list = value_iteration(env, gamma = (i+0.5)/10)
		policy = extract_policy(env,best_value, gamma = (i+0.5)/10)
		policy_score = evaluate_policy(env, policy, gamma=(i+0.5)/10, n=100)
		gamma = (i+0.5)/10
		plot = plot_policy_map('Frozen Lake Policy Map Iteration '+ str(i) + ' (VI) ' + 'Gamma: '+ str(gamma),policy.reshape(frozen_lake_size,frozen_lake_size),desc,colors_lake(),directions_lake())
		end=time.time()
		gamma_arr[i]=(i+0.5)/10
		iters[i]=k
		best_vals[i] = best_value
		list_scores[i]=np.mean(policy_score)
		time_array[i]=end-st
		rewards_list.append(np.array(score_list))
		value_diff_list.append(value_diff)


	plt.plot(gamma_arr, time_array)
	plt.xlabel('Gammas')
	plt.title(f'Frozen Lake-VI-Execution Time-{frozen_lake_size}*{frozen_lake_size}')
	plt.ylabel('Execution Time (s)')
	plt.grid()
	plt.show()

	plt.plot(gamma_arr,list_scores)
	plt.xlabel('Gammas')
	plt.ylabel('Average Rewards')
	plt.title(f'Frozen Lake-VI-Reward vs Gammas-{frozen_lake_size}*{frozen_lake_size}')
	plt.grid()
	plt.show()

	plt.plot(gamma_arr,iters)
	plt.xlabel('Gammas')
	plt.ylabel('Iterations to Converge')
	plt.title(f'Frozen Lake-VI-Convergence vs Gammas-{frozen_lake_size}*{frozen_lake_size}')
	plt.grid()
	plt.show()

	plt.plot(gamma_arr,best_vals) #, label=f'Gamma={gamma_arr[9]}')
	plt.xlabel('Gammas')
	plt.ylabel('Optimal Value')
	plt.title(f'Frozen Lake-VI-Best Value-{frozen_lake_size}*{frozen_lake_size}')
	plt.grid()
	plt.show()

	for g in range(len(gamma_arr)):
		plt.plot(value_diff_list[g], label=f'Gamma={gamma_arr[g]}')
	plt.legend(loc='best')
	plt.xlabel('Iterations')
	plt.ylabel('Value Difference')
	plt.title(f'Frozen Lake-VI-Value Difference vs Iterations-{frozen_lake_size}*{frozen_lake_size}')
	plt.grid()
	plt.show()

	for g in range(len(gamma_arr)):
		plt.plot(rewards_list[g], label=f'Gamma={gamma_arr[g]}')
	# plt.plot(score_list)
	plt.legend(loc='best')
	plt.xlabel('Iterations')
	plt.ylabel('Average Rewards')
	plt.title(f'Frozen Lake-VI-Rewards vs Iterations-{frozen_lake_size}*{frozen_lake_size}')
	plt.grid()
	plt.show()


	### Q-LEARNING #####
	print('Q LEARNING FOR FROZEN LAKE')
	st = time.time()
	reward_array = []
	iter_array = []
	size_array = []
	chunks_array = []
	averages_array = []
	time_array = []
	Q_array = []
	for epsilon in epsilon_list:
		Q = np.zeros((env.observation_space.n, env.action_space.n))
		rewards = []
		iters = []
		optimal=[0]*env.observation_space.n
		alpha = 0.85
		gamma = 0.95
		episodes = 10000
		environment  = 'FrozenLake-v1'
		env = gym.make(environment,desc=generate_random_map(size=frozen_lake_size))
		env = env.unwrapped
		desc = env.unwrapped.desc
		for episode in range(episodes):
			state = env.reset()[0] # observation
			done = False
			t_reward = 0
			max_steps = q_max_iter
			for i in range(max_steps):
				if done:
					break
				current = state
				if np.random.rand() < (epsilon):  # epsilon is exploration exploitation factor
					action = np.argmax(Q[current, :])
				else:
					action = env.action_space.sample()
				
				state, reward, done, _ ,_ = env.step(action)
				t_reward += reward
				Q[current, action] += alpha * (reward + gamma * np.max(Q[state, :])-Q[current, action])
			epsilon=(1-math.e**(-episode/1000))
			rewards.append(t_reward)
			iters.append(i)


		for k in range(env.observation_space.n):
			optimal[k]=np.argmax(Q[k, :])

		reward_array.append(rewards)
		iter_array.append(iters)
		Q_array.append(Q)

		env.close()
		end=time.time()
		#print("time :",end-st)
		time_array.append(end-st)

		# Plot results
		def chunk_list(l, n):
			for i in range(0, len(l), n):
				yield l[i:i + n]

		size = int(episodes / 50)
		chunks = list(chunk_list(rewards, size))
		averages = [sum(chunk) / len(chunk) for chunk in chunks]
		size_array.append(size)
		chunks_array.append(chunks)
		averages_array.append(averages)


	for i in range(len(epsilon_list)):
		# plt.plot(range(0, len(reward_array[0]), size_array[0]), averages_array[0],label='epsilon=0.05')
		# plt.plot(range(0, len(reward_array[1]), size_array[1]), averages_array[1],label='epsilon=0.15')
		# plt.plot(range(0, len(reward_array[2]), size_array[2]), averages_array[2],label='epsilon=0.25')
		# plt.plot(range(0, len(reward_array[3]), size_array[3]), averages_array[3],label='epsilon=0.50')
		# plt.plot(range(0, len(reward_array[4]), size_array[4]), averages_array[4],label='epsilon=0.75')
		# plt.plot(range(0, len(reward_array[5]), size_array[5]), averages_array[5],label='epsilon=0.95')
		plt.plot(range(0, len(reward_array[i]), size_array[i]), averages_array[i],label=f'epsilon={epsilon_list[i]}')

	plt.legend()
	plt.xlabel('Iterations')
	plt.grid()
	plt.title(f'Frozen Lake-Q Learning-Constant Epsilon-{frozen_lake_size}*{frozen_lake_size}')
	plt.ylabel('Average Reward')
	plt.show()

	plt.plot(epsilon_list,time_array)
	plt.xlabel('Epsilon Values')
	plt.grid()
	plt.title(f'Frozen Lake-Q Learning-{frozen_lake_size}*{frozen_lake_size}')
	plt.ylabel('Execution Time (s)')
	plt.show()

	# axs[0,0].plot(Q_array[0])
	# axs[0,1].plot(Q_array[1])
	# axs[0,2].plot(Q_array[2])
	# axs[0,3].plot(Q_array[3])
	# axs[0,4].plot(Q_array[4])
	# axs[0,5].plot(Q_array[5])
	# plt.subplots_adjust(left=0.2,
	# 				bottom=0.2,
	# 				right=0.8,
	# 				top=0.8,
	# 				wspace=0.6,
	# 				hspace=0.6)

	for i in range(len(epsilon_list)):
		plt.subplot(1,6,i+1)
		plt.imshow(Q_array[i])
		plt.title(f'Epsilon={epsilon_list[i]} Size={frozen_lake_size}*{frozen_lake_size}')

		# plt.subplot(1,6,2)
		# plt.title(f'Epsilon=0.15')
		# plt.imshow(Q_array[1])

		# plt.subplot(1,6,3)
		# plt.title(f'Epsilon=0.25')
		# plt.imshow(Q_array[2])

		# plt.subplot(1,6,4)
		# plt.title(f'Epsilon=0.50')
		# plt.imshow(Q_array[3])

		# plt.subplot(1,6,5)
		# plt.title(f'Epsilon=0.75')
		# plt.imshow(Q_array[4])

		# plt.subplot(1,6,6)
		# plt.title(f'Epsilon=0.95')
		# plt.imshow(Q_array[5])
	plt.colorbar()
	plt.show()

def Taxi_Experiments():
	environment  = 'Taxi-v3'
	env = gym.make(environment)
	env = env.unwrapped
	desc = env.unwrapped.desc
	time_array=[0]*10
	gamma_arr=[0]*10
	iters=[0]*10
	list_scores=[0]*10

	### POLICY ITERATION TAXI: ####
	print('POLICY ITERATION FOR TAXI')
	for i in range(3,10):
		st=time.time()
		best_policy,k,policy_diff = policy_iteration(env, gamma = (i+0.5)/10)
		scores = evaluate_policy(env, best_policy, gamma = (i+0.5)/10)
		end=time.time()
		gamma_arr[i]=(i+0.5)/10
		list_scores[i]=np.mean(scores)
		iters[i]=k
		time_array[i]=end-st

	'''
	plt.plot(gamma_arr, time_array)
	plt.xlabel('Gammas')
	plt.title(f'Taxi-PI-Execution Time')
	plt.ylabel('Execution Time (s)')
	plt.grid()
	plt.show()

	plt.plot(gamma_arr,list_scores)
	plt.xlabel('Gammas')
	plt.ylabel('Average Rewards')
	plt.title(f'Taxi-PI-Reward')
	plt.grid()
	plt.show()

	plt.plot(gamma_arr,iters)
	plt.xlabel('Gammas')
	plt.ylabel('Iterations to Converge')
	plt.title(f'Taxi-PI-Convergence')
	plt.grid()
	plt.show()
	'''

	#### VALUE ITERATION TAXI: #####
	print('VALUE ITERATION FOR TAXI') 
	for i in range(2,10):
		print(i)
		st = time.time()
		best_value,k = value_iteration(env, gamma=(i+0.5)/10);
		policy = extract_policy(env, best_value, gamma=(i+0.5)/10)
		policy_score = evaluate_policy(env, policy, gamma=(i+0.5)/10, n=1000)
		end = time.time()
		gamma_arr[i]=(i+0.5)/10
		iters[i]=k
		time_array[i]=end-st
		print(policy)

	plt.plot(gamma_arr, time_array)
	plt.xlabel('Gammas')
	plt.title(f'Taxi-VI-Execution Time')
	plt.ylabel('Execution Time (s)')
	plt.grid()
	plt.show()

	plt.plot(gamma_arr,list_scores)
	plt.xlabel('Gammas')
	plt.ylabel('Average Rewards')
	plt.title(f'Taxi-VI-Reward')
	plt.grid()
	plt.show()

	plt.plot(gamma_arr,iters)
	plt.xlabel('Gammas')
	plt.ylabel('Iterations to Converge')
	plt.title(f'Taxi-VI-Convergence')
	plt.grid()
	plt.show()

	print('Q LEARNING FOR TAXI')
	st=time.time()
	Q = np.zeros((env.observation_space.n, env.action_space.n))
	rewards = []
	iters = []
	optimal=[0]*env.observation_space.n
	alpha = 1.0
	gamma = 1.0
	episodes = 10000
	epsilon=0
	for episode in range(episodes):
		state = env.reset()
		done = False
		t_reward = 0
		max_steps = q_max_iter
		for i in range(max_steps):
			if done:
				break
			current = state
			if np.random.rand()<epsilon:
				action = np.argmax(Q[current, :])
			else:
				action = env.action_space.sample()
			state, reward, done, info = env.step(action)
			t_reward += reward
			Q[current, action] += alpha * (reward + gamma * np.max(Q[state, :])-Q[current, action])
		epsilon=(1-math.e**(-episode/1000))
		alpha=math.e**(-episode/1000)
		rewards.append(t_reward)
		iters.append(i)
	for k in range(env.observation_space.n):
		optimal[k]=np.argmax(Q[k, :])
		if np.argmax(Q[k, :])==5:
			print(k)
	print(optimal)
	print("average :",np.average(rewards[2000:]))
	env.close()
	end=time.time()
	print("time :",end-st)
	def chunk_list(l, n):
		for i in range(0, len(l), n):
			yield l[i:i + n]

	size = 5
	chunks = list(chunk_list(rewards, size))
	averages = [sum(chunk) / len(chunk) for chunk in chunks]
	plt.plot(range(0, len(rewards), size)[200:], averages[200:])
	plt.xlabel('iters')
	plt.ylabel('Average Reward')
	plt.title(f'Taxi-Q Learning')
	plt.show()

def run_episode(env, policy, gamma, render = True):
	obs = env.reset()
	total_reward = 0
	step_idx = 0
	while True:
		if render:
			env.render()
		obs, reward, done , _, _ = env.step(int(policy[obs]))
		total_reward += (gamma ** step_idx * reward)
		step_idx += 1
		if done:
			break
	return total_reward

def evaluate_policy(env, policy, gamma , n = 100):
	scores = [run_episode(env, policy, gamma, False) for _ in range(n)]
	return np.mean(scores)

def extract_policy(env,v, gamma):
	policy = np.zeros(env.nS)
	for s in range(env.nS):
		q_sa = np.zeros(env.nA)
		for a in range(env.nA):
			q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in  env.P[s][a]])
		policy[s] = np.argmax(q_sa)
	return policy

def compute_policy_v(env, policy, gamma):
	v = np.zeros(env.nS)
	eps = 1e-5
	while True:
		prev_v = np.copy(v)
		for s in range(env.nS):
			policy_a = policy[s]
			v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, is_done in env.P[s][policy_a]])
		if (np.sum((np.fabs(prev_v-v))) <= eps):
			break
	return v

def policy_iteration(env, gamma):
	policy = np.random.choice(env.nA, size=(env.nS))  
	max_iters = policy_max_iter
	desc = env.unwrapped.desc
	policy_diff = [len(policy)]
	reward_list = [0]
	for i in range(max_iters):
		old_policy_v = compute_policy_v(env, policy, gamma)
		new_policy = extract_policy(env,old_policy_v, gamma)
		#if i % 2 == 0:
		#	plot = plot_policy_map('Frozen Lake Policy Map Iteration '+ str(i) + 'Gamma: ' + str(gamma),new_policy.reshape(frozen_lake_size,frozen_lake_size),desc,colors_lake(),directions_lake())
		#	a = 1
		test = policy == new_policy
		diff = len(test)-np.sum(test)
		policy_diff.append(diff)
		mean_score = evaluate_policy(env, new_policy, gamma)
		reward_list.append(mean_score)
		if (np.all(test)):
			k=i+1
			break
		policy = new_policy

	return policy, k, policy_diff, reward_list

def value_iteration(env, gamma):
	v = np.zeros(env.nS)  # initialize value-function
	max_iters = value_max_iter
	eps = 1e-10
	desc = env.unwrapped.desc
	value_diff = []
	score_list = []
	for i in range(max_iters):
		prev_v = np.copy(v)
		for s in range(env.nS):
			q_sa = [sum([p*(r + gamma*prev_v[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(env.nA)] 
			v[s] = max(q_sa)
		#if i % 50 == 0:
		#	plot = plot_policy_map('Frozen Lake Policy Map Iteration '+ str(i) + ' (VI) ' + 'Gamma: '+ str(gamma),v.reshape(frozen_lake_size,frozen_lake_size),desc,colors_lake(),directions_lake())
		diff = np.sum(np.fabs(prev_v-v))
		value_diff.append(diff)
		score_list.append(np.mean(v))
		if (np.sum(diff) <= eps):
			k=i+1
			break
	return v,k,value_diff,score_list

def plot_policy_map(title, policy, map_desc, color_map, direction_map):
	# fig = plt.figure()
	M = policy.shape[1]
	N = policy.shape[0]

	fig = plt.figure()
	ax = fig.add_subplot(111, xlim=(0, M), ylim=(0, N))
	font_size = 'x-large'
	if M > 16:
		font_size = 'small'
	for i in range(N):
		for j in range(M):
			y = N-i-1
			x = j
			p = plt.Rectangle([x, y], 1, 1)
			p.set_facecolor(color_map[map_desc[i,j]])
			ax.add_patch(p)

			text = ax.text(x+0.5, y+0.5, direction_map[policy[i, j]], weight='bold', size=font_size,
						   horizontalalignment='center', verticalalignment='center', color='w')
			

	plt.axis('off')
	plt.xlim((0, M))
	plt.ylim((0, N))
	plt.title(title)
	plt.tight_layout()
	plt.savefig(f'{title}.png')
	plt.close()

	return (plt)



def Forest_Experiments(forest_size):
	
	print('POLICY ITERATION FOR FOREST MANAGEMENT')
	P, R = mdptoolbox.example.forest(S=forest_size, p = 0.01)
	value_f = [0]*10
	policy = [0]*10
	iters = [0]*10
	time_array = [0]*10
	gamma_arr = [0] * 10
	# rew_array = []
	for i in range(0,10):
		pi = mdptoolbox.mdp.PolicyIteration(P, R, (i+0.5)/10)
		pi.run()
		gamma_arr[i]=(i+0.5)/10
		value_f[i] = np.mean(pi.V)
		policy[i] = pi.policy
		iters[i] = pi.iter
		time_array[i] = pi.time
		# rew_array.append(pi.reward_array)


	plt.plot(gamma_arr, time_array)
	plt.xlabel('Gammas')
	plt.title(f'Forest Management-PI-Execution Time-Size={forest_size}')
	plt.ylabel('Execution Time (s)')
	plt.grid()
	plt.show()

	
	plt.plot(gamma_arr,value_f)
	plt.xlabel('Gammas')
	plt.ylabel('Average Rewards')
	plt.title(f'Forest Management-PI-Reward-Size={forest_size}')
	plt.grid()
	plt.show()

	plt.plot(gamma_arr,iters)
	plt.xlabel('Gammas')
	plt.ylabel('Iterations to Converge')
	plt.title(f'Forest Management-PI-Convergence-Size={forest_size}')
	plt.grid()
	plt.show()


	print('VALUE ITERATION FOR FOREST MANAGEMENT')
	P, R = mdptoolbox.example.forest(S=forest_size)
	value_f = [0]*10
	policy = [0]*10
	iters = [0]*10
	time_array = [0]*10
	gamma_arr = [0] * 10
	# rew_array = []

	for i in range(0,10):
		pi = mdptoolbox.mdp.ValueIteration(P, R, (i+0.5)/10)
		pi.run()
		gamma_arr[i]=(i+0.5)/10
		value_f[i] = np.mean(pi.V)
		policy[i] = pi.policy
		iters[i] = pi.iter
		time_array[i] = pi.time
		# rew_array.append(pi.reward_array)

	plt.plot(gamma_arr, time_array)
	plt.xlabel('Gammas')
	plt.title(f'Forest Management-VI-Execution Time-Size={forest_size}')
	plt.ylabel('Execution Time (s)')
	plt.grid()
	plt.show()
	
	plt.plot(gamma_arr,value_f)
	plt.xlabel('Gammas')
	plt.ylabel('Average Rewards')
	plt.title(f'Forest Management-VI-Reward-Size={forest_size}')
	plt.grid()
	plt.show()

	plt.plot(gamma_arr,iters)
	plt.xlabel('Gammas')
	plt.ylabel('Iterations to Converge')
	plt.title(f'Forest Management-VI-Convergence-Size={forest_size}')
	plt.grid()
	plt.show()
	
	print('Q LEARNING FOR FOREST MANAGEMENT')
	P, R = mdptoolbox.example.forest(S=forest_size,p=0.1)  ## changing p will change the optimal policy
	value_f = []
	policy = []
	iters = []
	time_array = []
	Q_table = []
	rew_array = []
	for epsilon in epsilon_list:
		st = time.time()
		pi = mdp.QLearning(P,R,discount=0.95,n_iter=q_max_iter) # discount factor 0.95
		end = time.time()
		pi.run(epsilon)
		rew_array.append(pi.reward_array)
		value_f.append(np.mean(pi.V))
		policy.append(pi.policy)
		time_array.append(end-st)
		Q_table.append(pi.Q)
	

	for i in range(len(epsilon_list)):
		plt.plot(range(0,q_max_iter), rew_array[i],label=f'epsilon={epsilon_list[i]}')
		# plt.plot(range(0,10000), rew_array[1],label='epsilon=0.15')
		# plt.plot(range(0,10000), rew_array[2],label='epsilon=0.25')
		# plt.plot(range(0,10000), rew_array[3],label='epsilon=0.50')
		# plt.plot(range(0,10000), rew_array[4],label='epsilon=0.75')
		# plt.plot(range(0,10000), rew_array[5],label='epsilon=0.95')
	
	plt.legend(loc='best')
	plt.xlabel('Iterations')
	plt.grid()
	plt.title(f'Forest Mgt-Q Learning-Reward vs Iters-Size={forest_size}')
	plt.ylabel('Average Reward')
	plt.show()


	plt.figure(figsize=(6, 6))

	for i in range(len(epsilon_list)):
		plt.subplot(1,6,i+1)
		plt.title(f'Epsilon={epsilon_list[i]} Size={forest_size}')
		plt.imshow(Q_table[i][:20,:])

	# plt.subplot(1,6,1)
	# plt.imshow(Q_table[0][:20,:])
	# plt.title(f'Epsilon=0.05')

	# plt.subplot(1,6,2)
	# plt.title(f'Epsilon=0.15')
	# plt.imshow(Q_table[1][:20,:])

	# plt.subplot(1,6,3)
	# plt.title(f'Epsilon=0.25')
	# plt.imshow(Q_table[2][:20,:])

	# plt.subplot(1,6,4)
	# plt.title(f'Epsilon=0.50')
	# plt.imshow(Q_table[3][:20,:])

	# plt.subplot(1,6,5)
	# plt.title(f'Epsilon=0.75')
	# plt.imshow(Q_table[4][:20,:])

	# plt.subplot(1,6,6)
	# plt.title(f'Epsilon=0.95')
	# plt.imshow(Q_table[5][:20,:])
	plt.colorbar()
	plt.show()

	return

def colors_lake():
	return {
		b'S': 'green',
		b'F': 'skyblue',
		b'H': 'black',
		b'G': 'gold',
	}

def directions_lake():
	return {
		3: '⬆',
		2: '➡',
		1: '⬇',
		0: '⬅'
	}

# def actions_taxi():
# 	return {
# 		0: '⬇',
# 		1:'⬆',
# 		2: '➡',
# 		3: '⬅',
# 		4: 'P',
# 		5: 'D'
# 	}

# def colors_taxi():
# 	return {
# 		b'+': 'red',
# 		b'-': 'green',
# 		b'R': 'yellow',
# 		b'G': 'blue',
# 		b'Y': 'gold'
# 	}


#%%

q_max_iter= 50000
value_max_iter = 50000
policy_max_iter = 50000
epsilon_list = [0.7, 0.8, 0.9]  # 0.05,0.15,0.25

print('STARTING EXPERIMENTS')
# frozen_lake_size = 5
frozen_lake_size = 20

Frozen_Lake_Experiments(frozen_lake_size)


#%%

q_max_iter= 100000
value_max_iter = 100000
policy_max_iter = 100000
epsilon_list = [0.5, 0.75, 0.9]  # 0.05,0.15,0.25

forest_size = 50
# forest_size = 1000
Forest_Experiments(forest_size)
#Taxi_Experiments()
print('END OF EXPERIMENTS')




# %%
import gymnasium as gym
environment  = 'FrozenLake-v1'
env = gym.make(environment)
x = env.reset()
state, reward, done, _, _ = env.step(2)

# %%
