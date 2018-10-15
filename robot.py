import numpy as np 
import pdb
import random
import matplotlib
import time
from copy import deepcopy
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import argparse

def parse_args():
	"""
	Parse input arguments
	"""
	parser = argparse.ArgumentParser(description='Homework for Markov Decesion Process')
	parser.add_argument('--errorPr', dest='errorPr',help='probability of prerotating error',default=0.0, type=float)
	parser.add_argument('--dis', dest='dis', help='discount_factor',default=0.9, type=float)
	parser.add_argument('--initial', dest='initial', help='policy initialization', action='store_true')
	parser.add_argument('--plot', dest='plot', help='plot trajectory', action='store_true')
	parser.add_argument('--feature', dest='feature', help='type of iterations',
						default='policy_iteration', type=str)
	parser.add_argument('--mod', dest='mod',help='use modified reward function, which require pointing downward',action='store_true')
	args = parser.parse_args()
	return args

class robot:
	# Problem 1 a)
	def __init__(self, errorPr=0, discount=1):
		self.errorPr = errorPr
		self.discount = discount
		# state: [y, x, head]
		# action ['F', '+']: 'F' for forward, 'B' for backward, '0' for still, '+' for rotating clockwise, '-' for rotating anticlockwise
		self.actionSpace = [['F', '+'], ['F', '0'], ['F', '-'],
							['B', '+'], ['B', '0'], ['B', '-'],
							['0', '0']]
		self.probMatrix = np.array([[0.0]*432 for i in range(432)])
		#self.row, self.column = 6, 6
		self.reward = [[-100, -100, -100, -100, -100, -100],
					   [-100, 0,	0,	  0, 	0,	  -100],
					   [-100, 0,	-10,  0,	-10,  -100],
					   [-100, 0,	-10,  0,	-10,  -100],
					   [-100, 0,	-10,  1,	-10,  -100],
					   [-100, -100, -100, -100, -100, -100],
						]

		self.valueMatrix = np.array([[[0.0 for _ in range(12)] for _ in range(6)] for _ in range(6)])
		self.actionMatrix = [[[['0', '0'] for _ in range(12)] for _ in range(6)] for _ in range(6)]

	def computeNextState(self, currentState, action, prerotateError=True):
		# Problem 1 d)
		# prerotateError: to manually control whether the next state is deterministic
		# currentState & nextState: [y, x, head]
		# action ['F', '+']: 'F' for forward, 'B' for backward, '0' for still, '+' for clockwise, '-' anticlockwise
		if action[0] == '0' and action[1] == '0': return currentState
		if prerotateError:
			temp = random.uniform(0, 1)
			if temp < self.errorPr:
				preError = '-'
				currentState[2] = (currentState[2]-1)%12
			elif temp < 2*self.errorPr:
				preError = '+'
				currentState[2] = (currentState[2]+1)%12
			else:
				preError = '0'

		up, down, right, left = set([11,0,1]), set([7,6,5]), set([2,3,4]), set([8,9,10])

		nextState = currentState[:]

		# check the heading of robot, and then take actions correspondingly
		if currentState[2] in up:
			nextState[0] = currentState[0]+1 if action[0] == 'F' else currentState[0]-1
		elif currentState[2] in down:
			nextState[0] = currentState[0]-1 if action[0] == 'F' else currentState[0]+1
		elif currentState[2] in right:
			nextState[1] = currentState[1]+1 if action[0] == 'F' else currentState[1]-1
		else:
			nextState[1] = currentState[1]-1 if action[0] == 'F' else currentState[1]+1
		
		if action[1] == '+':
			nextState[2] = (nextState[2]+1)%12 
		elif action[1] == '-':
			nextState[2] = (nextState[2]-1)%12
		
		# clip the next status beween [0, 5]
		for i in range(2):
			if nextState[i] < 0:
				nextState[i] = 0
			elif nextState[i] > 5:
				nextState[i] = 5

		return nextState

	def computeNextStateList(self, currentState, action):
		# given current state and action, return all possible state after taking the action
		result = []
		if action[0] == '0' and action[1] == '0':
			return [currentState]
		else:
			result.append(self.computeNextState(currentState, action, False))
			temp = currentState[:]
			temp[2] = (temp[2]-1)%12
			result.append(self.computeNextState(currentState, action, False))
			temp = currentState[:]
			temp[2] = (temp[2]+1)%12
			result.append(self.computeNextState(currentState, action, False))
			return result	


	def probActionState(self, currentState, nextState, action):
		# Problem 1 d)
		# return the probabiltiy of the next state given the current state and actions
		# currentState & nextState: [y, x, head]
		# action ('F', '+'): 'F' for forward, 'B' for backward, '0' for still, '+' for clockwise, '-' anticlockwise
		if action[0] == '0' and action[1] == '0':
			return 1 if nextState == currentState else 0

		if abs(currentState[0]-nextState[0])>1 or abs(currentState[1]-nextState[1])>1:
			return 0
		else:
			intentNextState = self.computeNextState(currentState, action, False)
			temp = currentState[:]
			temp[2] = (temp[2]+1)%12
			errorNextState1 = self.computeNextState(temp, action, False)
			temp = currentState[:]
			temp[2] = (temp[2]-1)%12
			errorNextState2 = self.computeNextState(temp, action, False)

			# return probability correspondingly
			if intentNextState == nextState:
				return 1-2*self.errorPr
			elif nextState == errorNextState1 or nextState == errorNextState2:
				return self.errorPr
			else:
				return 0

	def getReward(self, state):
		# Problem 2 a)
		# return reward given the topology of environment
		return self.reward[state[0], state[1]]

	def getConditionalReward(self, state):
		# return reward given the topology of environment, under the condition that
		# robot has to head downward
		down = set([5,6,7])
		return 1.0 if state[0]==4 and state[1]==3 and state[2] in down else 0.0

	def initialPolicy(self):
		# Problem 3 a)
		# flood the actionMatrix with initial policy
		# goal position: idx (y, x) = (4, 3)
		def computeDistance(currentState, nextState):
			return sum([abs(currentState[i]-nextState[i]) for i in range(2)])

		up, down, right, left = set([11,0,1]), set([7,6,5]), set([2,3,4]), set([8,9,10])
		for y in range(6):
			for x in range(6):
				for h in range(12):
					currentAction = ['0', '0']
					if y == 4 and x == 3:
						#pdb.set_trace()
						self.actionMatrix[y][x][h] = currentAction
						continue
					# check whether the goal is in front of the robot
					# if in front, then move forward; otherwise then move backward
					if h in up:
						currentAction[0] = 'F' if y < 4 else 'B'
					elif h in down:
						currentAction[0] = 'F' if y > 4 else 'B'
					elif h in right:
						currentAction[0] = 'F' if x < 3 else 'B'
					else:
						currentAction[0] = 'F' if x > 3 else 'B'
					#tempNextState = self.computeNextState([y, x, h], currentAction, prerotateError = False)

					# after moving forward, determine rotating direction based on the distance of the next action to the goal
					action1, action2, action3 = [currentAction[0], '+'], [currentAction[0], '0'], [currentAction[0], '-']
					nextState1, nextState2, nextState3 = self.computeNextState([y, x, h], action1, prerotateError = False), \
														self.computeNextState([y, x, h], action2, prerotateError = False), \
														self.computeNextState([y, x, h], action3, prerotateError = False)
					actionList = [action1, action1, action2, action2, action3, action3]
					stateList = [self.computeNextState(nextState1, ['F', '0'], prerotateError=False),
								 self.computeNextState(nextState1, ['B', '0'], prerotateError=False),
								 self.computeNextState(nextState2, ['F', '0'], prerotateError=False),
								 self.computeNextState(nextState2, ['B', '0'], prerotateError=False),
								 self.computeNextState(nextState3, ['F', '0'], prerotateError=False),
								 self.computeNextState(nextState3, ['B', '0'], prerotateError=False),]

					List = [[computeDistance([4, 3], stateList[i][:2]), actionList[i]] for i in range(6)]
					List = sorted(List, key = lambda x:x[0])
					#if y == 0 and x == 0 and h == 3:
						#pdb.set_trace()
					self.actionMatrix[y][x][h] = List[0][1]

	def getTrajectory(self, startPoint, actionMatrix):
		# Problem 3 b)
		result = []
		idx = startPoint[:]
		result.append(idx)
		i = 0
		print('trajectory:')
		while idx[0] != 4 or idx[1] != 3:
			idx = self.computeNextState(idx, actionMatrix[idx[0]][idx[1]][idx[2]], prerotateError=False)
			if idx in result: break
			result.append(idx)
			print(idx)
		return result
	
	
	def plotTrajectory(self, trajectory):
		# Problem 3 b)
		tra = trajectory
		plt.plot([t[1] for t in tra], [t[0] for t in tra])
		plt.ylabel('y-axis')
		plt.xlabel('x-axis')
		plt.xlim(0,5)
		plt.ylim(0,5)
		plt.title('trajectory of an action')
		plt.grid()
		plt.show()

	def computeValue(self, iteration=20, modified=False):
		# Problem 3 d)
		# Iterative policy evaluation.
		# Iteration is the number of iterations conducted for one evaluation
		def isAjcent(currentState, nextState):
			return False if sum([abs(currentState[i]-nextState[i]) for i in range(2)]) > 1 \
					else True
		valueMatrix = self.valueMatrix[:]
		for _ in range(iteration):
			for i in range(6):
				for j in range(6):
					for k in range(12):
						# for a specific action given a state, compute its value given the old values from the last iteration
						currentState = [i, j, k]
						nextState = []
						action = self.actionMatrix[i][j][k]
						nextState.append(self.computeNextState([i, j, k], action, False))
						nextState.append(self.computeNextState([i, j, (k+1)%12], action, False))
						nextState.append(self.computeNextState([i, j, (k-1)%12], action, False))
						#if i == 4 and j == 0 and k == 0: pdb.set_trace()
						if not modified:
							self.valueMatrix[i][j][k] = sum([self.probActionState(currentState, state, action)*\
														valueMatrix[state[0]][state[1]][state[2]] \
														for state in nextState])*self.discount + self.reward[i][j]
						else:
							self.valueMatrix[i][j][k] = sum([self.probActionState(currentState, state, action)*\
														valueMatrix[state[0]][state[1]][state[2]] \
														for state in nextState])*self.discount + self.getConditionalReward(currentState)

	def computeValuePesudoInverse(self):
		# another method for value evaluation by solving equations
		def isAjcent(currentState, nextState):
			return False if sum([abs(currentState[i]-nextState[i]) for i in range(2)]) > 1 \
					else True
		#x = np.resize(valueMatrix, (432, 1))
		idx = [[[[i, j, k] for k in range(12)] for j in range(6)] for i in range(6)]
		idx = np.resize(idx, (432, 3))
		idxDict = dict(zip([i for i in range(432)], idx))
		A = np.array([[0.0]*432 for i in range(432)])
		for i in range(len(idx)):
			for j in range(len(idx)):
				currentState, nextState = [idxDict[i][k] for k in range(3)], [idxDict[j][k] for k in range(3)]
				if isAjcent(currentState, nextState):			
					A[i][j] = self.probActionState(currentState, nextState, \
							self.actionMatrix[currentState[0]][currentState[1]][currentState[2]])*self.discount
				else:
					A[i][j] = 0.0
		self.probMatrix = A
		b = []
		for i in range(432):
			b.append(self.reward[idx[i][0]][idx[i][1]])
		
		x = np.dot(np.linalg.pinv(np.identity(432)-A), b)
		self.valueMatrix = np.resize(x, (6,6,12))


	def updatePolicy(self):
		# Problem 3 f)
		# traverse all possible actions of an input state, find out the optimal action that maximize value of the state
		updated = False
		for i in range(6):
			for j in range(6):
				for k in range(12):
					actionSpace = self.actionSpace[:]
					currentState = [i, j, k]
					pair = []
					#pdb.set_trace()
					for action in actionSpace:
						nextState = []
						nextState.append(self.computeNextState([i, j, k], action, False))
						nextState.append(self.computeNextState([i, j, (k+1)%12], action, False))
						nextState.append(self.computeNextState([i, j, (k-1)%12], action, False))
						
						value = ([self.probActionState(currentState, state, action)*\
							   self.valueMatrix[state[0]][state[1]][state[2]] for state in nextState])
						pair.append((sum(value), action))
						#if i == 2 and j == 1 and k == 9: pdb.set_trace()
					pair = sorted(pair, key=lambda x:x[0], reverse=True)
					if self.actionMatrix[i][j][k] != pair[0][1]:
						self.actionMatrix[i][j][k] = pair[0][1]
						updated = True
		return updated

	def policyIteration(self, iteration=20, modified=False):
		# Problem 3 g)
		# merge policy evaluation and policy searching as policy iteration
		self.initialPolicy()
		updated = True
		# stop the iteration given no more policy being updated
		while updated:
			self.computeValue(iteration, modified)
			updated = self.updatePolicy()

	#Problem 4a
	# valueIteration(horizon)
	# this function perform value iteration
	# horizon: limit the maximum iteration, in case of convergence never happened
	#
	# return valueMatrix, actionMatrix
	# valueMatrix: the value of each state
	# actionMatrix: policy matrix, each state corresponding to one action
	def valueIteration(self, horizon):
		#Problem 4a
		self.valueMatrix = np.zeros((6,6,12)) #reset valueMatrix to zeros
		for n in range(horizon): #iterate until meet horizon

			#assign update value to Q(s',a)
			valueHolder = deepcopy(self.valueMatrix)

			#Iterate through all Current State
			for i in range(6):
				for j in range(6):
					for k in range(12):

						actionValueCollection = [] #Hold 7 Q(s,a)

						#iterate through 7 action
						for a in range(7):
							currentState = [i,j,k]
							Qsa = 0.0
							nextState = []
							#possible next three state given action
							nextState.append(self.computeNextState([i, j, k], self.actionSpace[a], False))
							nextState.append(self.computeNextState([i, j, (k+1)%12], self.actionSpace[a], False))
							nextState.append(self.computeNextState([i, j, (k-1)%12], self.actionSpace[a], False))

							#Calculate Q(s,a)
							for state in nextState:
								Qsa += self.probActionState(currentState,state,self.actionSpace[a]) * \
										(float(self.reward[i][j])+ self.discount * valueHolder[state[0]][state[1]][state[2]])
							#Store Q(s,a)
							actionValueCollection.append(Qsa)
						Qsa = 0.0 #reset Qsa value

						#Compare Q(s,a1)~Q(s,a7), choose the action with highest Q(s,a)
						self.valueMatrix[i][j][k] = np.max(actionValueCollection)
						self.actionMatrix[i][j][k] = self.actionSpace[actionValueCollection.index(np.max(actionValueCollection))]

			#exist loop early if converge
			if np.all(self.valueMatrix.round(0) == valueHolder.round(0)):
				print( 'number of iteration {}'.format(n))
				break

		return self.valueMatrix.round(0), self.actionMatrix

if __name__ == '__main__':
	args = parse_args()
	modifiedReward = args.mod
	robot = robot(errorPr=args.errorPr, discount=args.dis)
	if args.initial:
		robot.initialPolicy()
		result = robot.getTrajectory([4,1,6], robot.actionMatrix)
		if args.plot:
			robot.plotTrajectory(result)	

	if args.feature == 'policy_iteration':
		a = time.time()
		robot.policyIteration(iteration=20, modified=modifiedReward)
		b = time.time()
		print('Time comsumed of policy iteration: {:.3f}'.format(b-a))
		result = robot.getTrajectory([4,1,6], robot.actionMatrix)
		if args.plot:
			robot.plotTrajectory(result)

	elif args.feature == 'value_iteration':
		a = time.time()
		robot.valueIteration(1000) #horizon set to 1000
		b= time.time()
		print('Time comsume of value iteration: {:.3f}'.format(b-a))
		result = robot.getTrajectory([4,1,6], robot.actionMatrix)
		if args.plot:
			robot.plotTrajectory(result)
	else:
		print('Unrecognized iteration type.')
		pdb.set_trace()

