#
# usage:
#
# python dqn.py x show x
#
# python dqn.py
#
# python dqn.py load show save
#

import random
import numpy as np
import numpy
import pprint
import itertools
import sys
import cPickle
import theano
import theano.tensor as T
import pygame
from pygame.locals import *
import datetime
import re

rng = np.random.RandomState()

# utility functions

def coin(): return rng.uniform() > 0.5
def normalize(x): return (x-np.mean(x))/np.std(x)

# block game

kinds = [ # standard pieces
		[[0,11,0],[11,11,11],[0,0,0]],
		[[0,0,12],[12,12,12],[0,0,0]],
		[[0,0,0,0],[13,13,13,13],[0,0,0,0],[0,0,0,0]],
		# [[14,14,0],[14,14,0],[0,0,0]], # rotatable 2x2 
		[[14,14],[14,14]], # non-rotatable 2x2
		[[15,15,0],[0,15,15],[0,0,0]],
		[[16,0,0],[16,16,16],[0,0,0]],
		[[0,17,17],[17,17,0],[0,0,0]]
		]

def g_standard(): # standard piece generator
	while True:
		seq = numpy.random.permutation(7)
		for i in xrange(7):
			yield kinds[seq[i]]	

def rcf(): return 11+np.random.randint(7) # random color function

def g_curriculum(): # curriculum learning
	while True:
		for j in xrange(50):
			seq = numpy.random.permutation(7)
			for i in xrange(7): yield kinds[seq[i]]
		# for i in xrange(7*4): yield [[18]] # achieve random exploration through non-random optimism
		for j in xrange(100):
			yield np.array([[rcf()]])

# choose piece generator and set first piece

# g = g_standard()
g = g_curriculum()
p = g.next()
next_p = g.next()
p_x = 7
p_y = 0

b_l = 20
b_w = 10

board = [[0,0,10]+[0 for x in xrange(b_w)]+[10,0,0] for y in xrange(b_l)] \
		+[[0,0]+[10 for x in xrange(b_w+2)]+[0,0]] \
		+[[0 for x in xrange(b_w+6)]] \
		+[[0 for x in xrange(b_w+6)]]

def rotate_array(a):
	return numpy.array(zip(*a[::-1]))

def add_p(x,y):
	for i,j in itertools.product(xrange(len(p)),xrange(len(p[0]))):
		board[y+j][x+i] += p[i][j]

def remove_p(x,y):
	for i,j in itertools.product(xrange(len(p)),xrange(len(p[0]))):
		board[y+j][x+i] -= p[i][j]

def collision():
	for i in xrange(b_l+3):
		for j in xrange(b_w+6):
			if board[i][j] > 20:
				return True
	return False

def rotate_p():
	global p
	remove_p(p_x,p_y)
	p = rotate_array(p)
	add_p(p_x,p_y)

	if collision(): # unrotate

		remove_p(p_x,p_y)
		p = rotate_array(p)
		p = rotate_array(p)
		p = rotate_array(p)
		add_p(p_x,p_y)

def clear_filled_rows():
	global reward_2, line_clears
	for i in xrange(b_l-1,-1,-1):
		full = True
		for j in xrange(3,b_w+3,1):
			if board[i][j] == 0:
				full = False
		if full:
			print 'line clear'
			reward_2 += 0.01 # reinforce
			line_clears += 1
			if show: animate_line_clear() # it's weird that nn doesn't get to see this
			for j in xrange(3,b_w+3,1):
				board[i][j] = 0
			for k in xrange(i,0,-1):
				for j in xrange(3,b_w+3,1):
					board[k][j] = board[k-1][j]
			clear_filled_rows()

def clear_board():
	for i in xrange(b_l-1,-1,-1):
		for j in xrange(3,b_w+3,1):
			board[i][j] = 0

def move_p_down():
	global p, p_x, p_y, next_p
	remove_p(p_x,p_y)
	p_y += 1
	add_p(p_x,p_y)

	if collision():

		# uncollide
		remove_p(p_x,p_y)
		p_y -= 1
		add_p(p_x,p_y)

		clear_filled_rows()

		# spawn new p
		p_x = 7
		p_y = 0
		p = next_p
		next_p = g.next()
		add_p(p_x,p_y)
		if coin(): rotate_p() # spawn with random orientation

		return False # tell drop_p() to stop

	return True

def move_p_left():
	global p, p_x, p_y
	remove_p(p_x,p_y)
	p_x -= 1
	add_p(p_x,p_y)

	if collision():
		remove_p(p_x,p_y)
		p_x += 1
		add_p(p_x,p_y)

def move_p_right():
	global p, p_x, p_y
	remove_p(p_x,p_y)
	p_x += 1
	add_p(p_x,p_y)

	if collision():
		remove_p(p_x,p_y)
		p_x -= 1
		add_p(p_x,p_y)

def drop_p():
	while move_p_down():
		pass

add_p(p_x,p_y) # starting piece

# console view

def print_board(): print np.asarray(board)
def pretty_print_board(): 
	x = np.asarray([row[2:b_w+4] for row in board[0:b_l+1]])
	np.place(x,x>0,1)
	print re.sub('[\[\]0]',' ',re.sub('[1]','*',np.array_str(x)))

# pygame view

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255,255,0)
PURPLE = (255,0,255)
CYAN = (0,255,255)
PINK = (255,128,128)
ORANGE = (255,128,0)
DARK_GRAY = (64,64,64)

def color_map(n):
	if n == 0:
		return BLACK
	elif n == 10:
		return GRAY
	elif n == 11:
		return RED
	elif n == 12:
		return GREEN
	elif n == 13:
		return BLUE
	elif n == 14:
		return YELLOW
	elif n == 15:
		return PURPLE
	elif n == 16:
		return CYAN
	elif n == 17:
		return ORANGE
	elif n == 18:
		return DARK_GRAY
	else: 
		return PINK

def draw_board():
	for i in xrange(b_l+3): # board
		for j in xrange(b_w+6):
			pygame.draw.rect(windowSurface, color_map(board[i][j]), (100+j*20, 50+i*20, 19, 19))
	# for i in xrange(4):
		# for j in xrange(4):
			# pygame.draw.rect(windowSurface, BLACK, (450+j*20, 100+i*20, 19, 19))
	# for i in xrange(len(next_p)):
		# for j in xrange(len(next_p[0])):
			# pygame.draw.rect(windowSurface, color_map(next_p[i][j]), (450+j*20, 100+i*20, 19, 19))
	pygame.draw.rect(windowSurface, WHITE, (100+p_x*20,50+p_y*20,10,10)) # draw active location
	for i in xrange(21):
		for j in xrange(12):
			c = nn_view[i][j]
			pygame.draw.rect(windowSurface, (c,c,c), (420+j*20, 50+i*20, 19, 19))
	for i in xrange(21):
		for j in xrange(12):
			c = nn_mem_view[i][j]
			pygame.draw.rect(windowSurface, (c,c,c), (700+j*20, 50+i*20, 19, 19))


def flash_red():
	for i in xrange(b_l+3):
		for j in xrange(b_w+6):
			if board[i][j] != 0:
				pygame.draw.rect(windowSurface, RED, (100+j*20, 50+i*20, 19, 19))	

def flash_green():
	for i in xrange(b_l+3):
		for j in xrange(b_w+6):
			if board[i][j] != 0:
				pygame.draw.rect(windowSurface, GREEN, (100+j*20, 50+i*20, 19, 19))

def animate():
	draw_board()
	pygame.display.update()
	pygame.event.clear()

def animate_game_over():
	draw_board()
	flash_red()
	pygame.display.update()
	pygame.time.wait(flash_time)

def animate_line_clear():
	draw_board()
	flash_green()
	pygame.display.update()
	pygame.time.wait(flash_time)	

# dqn

# init/load model

# np.random.seed(0) # never forget
# def layer(n_in,n_out): # orthogonal initialization
# 	if n_in <= 1 or n_out <= 1: return theano.shared(value=np.asarray(rng.normal(loc=0.0, scale=0.5*1.0*np.sqrt(6.0/(n_in+n_out)), size=(n_in,n_out)), dtype=theano.config.floatX),name='W',borrow=True)
# 	U, s, V = np.linalg.svd(rng.normal(loc=0.0, scale=0.5*1.0*np.sqrt(6.0/(n_in+n_out)), size=(n_in,n_out)))
# 	return theano.shared(value=np.asarray(np.dot(U,np.dot(np.diag(np.ones(s.shape)),V)), dtype=theano.config.floatX),name='W',borrow=True)
def layer(n_in,n_out): return theano.shared(value=np.asarray(rng.normal(loc=0.0, scale=0.5*1.0*np.sqrt(6.0/(n_in+n_out)), size=(n_in,n_out)), dtype=theano.config.floatX),name='W',borrow=True)
# def layer(n_in,n_out): return theano.shared(value=np.asarray(rng.normal(loc=0.0, scale=1.0*np.sqrt(1.0/(n_in+n_out)), size=(n_in,n_out)), dtype=theano.config.floatX),name='W',borrow=True)
def bias(n_out): return theano.shared(value=np.zeros(n_out,dtype=theano.config.floatX),name='b',borrow=True)
# def positive_bias(n_out): return theano.shared(value=np.ones(n_out,dtype=theano.config.floatX),name='b',borrow=True)
def positive_bias(n_out): return theano.shared(value=np.array([100.0],dtype=theano.config.floatX),name='b',borrow=True) # optimism under uncertainty

m = 10 #3

if m%2: print 'number of layers must be even for resnet!'

if len(sys.argv) > 1 and sys.argv[1] == 'load':
	filename='model.save'
	print 'model', filename, 'selected'
	print 'loading model..'
	save_file = open(filename, 'rb') 
	# for param in params:
		# cPickle.dump(param, save_file, -1)
	# W0 = cPickle.load(save_file)
	# W1 = cPickle.load(save_file)
	# W2 = cPickle.load(save_file)
	# W3 = cPickle.load(save_file)
	# b0 = cPickle.load(save_file)
	# b1 = cPickle.load(save_file)
	# b2 = cPickle.load(save_file)
	# b3 = cPickle.load(save_file)
	# m = 10 #3
	W=[0]*(m+1)
	b=[0]*(m+1)
	for i in xrange(m+1):
		W[i] = cPickle.load(save_file)
	for i in xrange(m+1):
		b[i] = cPickle.load(save_file)
	WW = W
	bb = b
	save_file.close()
else:
	print 'initializing new model..'
	# m = 10 #3
	W=[0]*(m+1)
	b=[0]*(m+1)
	WW=[0]*(m+1)
	bb=[0]*(m+1)
	for i in xrange(m):
		W[i] = layer(256,256)
		b[i] = bias(256)
		WW[i] = layer(256,256)
		bb[i] = bias(256)
		# W[i] = layer(256,256)
		# b[i] = bias(256)
		# WW[i] = layer(256,256)
		# bb[i] = bias(256)
	W[m] = layer(256,1)
	b[m] = bias(1)
	WW[m] = layer(256,1)
	bb[m] = bias(1)
	# # b[m] = positive_bias(1)
	# W[0] = layer(256,256)
	# W[1] = layer(256,128)
	# W[2] = layer(128,64)
	# W[3] = layer(64,1)
	# b[0] = bias(256)
	# b[1] = bias(128)
	# b[2] = bias(64)
	# b[3] = bias(1)
	# b[3] = positive_bias(1)

# hyperparams and misc

gamma = 0.9 #0.5 #0.0 #0.5 #0.9 #0.9 # 0.995 # 0.99 # large gamma slows down degradation of initial optimism
epsilon = 0.0 #0.01 #0.5 #1.0 #0.3 #1.0 #0.1 #0.001 #0.001 # 0.05
flash_time = 200 # win/lose animation time
move_accounting = [0,0,0,0] # keep track of moves played
line_clears = 0
game_count = 0
happiness = 0 # display dqn happiness with brightness/vibrance
learning_rate = 0.001 #0.01
# learning_rate = 1.0/(204.0*float(m)) #not bad 0.0001 #0.01 #0.01 # smaller learning rate leads to more uniform move exploration
# r_sensitivity = 0.1 #1.0 # theoretically should be 1-gamma if max q is 1
novelty = 0 # remember memory burn in and use it to measure novelty
# mem_size = 100000
mem_size = 10000
memory = [0]*mem_size
priority = [0]*mem_size
max_priority = 100.0
init_watch_time = 5
watch_time = 0
score = 0.0 # score = 0 on game over, score += 1 on line

# theano/nn

y = T.matrix(dtype=theano.config.floatX)
X = T.matrix(dtype=theano.config.floatX)
def fun(x): return T.nnet.relu(x,alpha=0.1)
# fun = T.tanh
# fun = T.nnet.relu
dot = T.dot
accum = X
# replace block mapping x -> f(x+wf(wx)) with  x -> x+f(f(x)w)w
# for i in xrange(m):
	# accum = fun(dot(accum,W[i])+b[i])
for i in xrange(m):
	if not i%2:
		print i
		print i+1
		accum = accum + dot(fun(dot(fun(accum),W[i])+b[i]),W[i+1])+b[i+1]
print m
output = dot(accum,W[m])+b[m]
target_accum = X
for i in xrange(m):
	target_accum = fun(dot(target_accum,WW[i])+bb[i]) #+X)
target_output = dot(accum,WW[m])+bb[m]
# output = fun(dot(fun(dot(fun(dot(fun(dot(X,W[0])+b[0]),W[1])+b[1]),W[2])+b[2]),W[3])+b[3])
# output = dot(fun(dot(fun(dot(fun(dot(X,W[0])+b[0]),W[1])+b[1]),W[2])+b[2]),W[3])+b[3] # all of R
# output = T.tanh(dot(accum,W[m])+b[m])
cost = T.sum((y - output) ** 2)
# cost = T.sum((y - output) ** 1) # needs changes to prioritized mem to be used 
params = [W[i] for i in xrange(m+1)]+[b[i] for i in xrange(m+1)]
target_params = [WW[i] for i in xrange(m+1)]+[bb[i] for i in xrange(m+1)]
updates = [(param,param-learning_rate*gradient) for param,gradient in zip(params,T.grad(cost,params))]
target_updates = [(target_param,param) for target_param,param in zip(target_params,params)]
train = theano.function(inputs=[X,y], outputs=[output,cost], updates=updates)
test = theano.function(inputs=[X], outputs=[output])
target_test = theano.function(inputs=[X],outputs=[target_output])
target_update = theano.function(inputs=[],outputs=[],updates=target_updates)

# render games in pygame

show = False
if len(sys.argv) > 2 and sys.argv[2] == 'show': show = True

if show:
	print 'initializing pygame view..'
	pygame.init()
	windowSurface = pygame.display.set_mode((1200, 600), 0, 32)
	windowSurface.fill(BLACK)
	pygame.display.set_caption('standard dqn')

# q learning

def get_state(): # get state
	# return np.reshape([subl[3:b_w+3] for subl in board[0:b_l]],(1,200)) # no attention
	return np.reshape([subl[2:b_w+4] for subl in board[0:b_l+1]],(1,252)) # no attention but include walls and floor
	# return np.reshape(np.roll(np.roll(np.asarray([row[2:b_w+4] for row in board[0:b_l+1]]),-(p_x-7),axis=1),-p_y,axis=0),(1,252)) # attention
def get_state_action(state,action): 
	# return 0.1*normalize(np.array(np.concatenate((state,action),axis=1),dtype=float))
	# return 0.1*normalize(rng.normal(loc=0.0,scale=0.25,size=(1,256))+normalize(np.array(np.concatenate((state,action),axis=1),dtype=float))) # inject noise
	return 0.1*normalize(rng.normal(loc=0.0,scale=0.5,size=(1,256))+normalize(np.array(np.concatenate((state,action),axis=1),dtype=float))) # inject noise
def print_state(a_state): print np.reshape(a_state,(20+1,10+2))
def pretty_print_state(a_state):
	# x = np.asarray([row[2:b_w+4] for row in a_state[0:b_l+1]])
	x = np.reshape(a_state,(21,12))
	np.place(x,x>0,1)
	print re.sub('[\[\]0]',' ',re.sub('[1]','*',np.array_str(x)))
def print_state_action(a_state_action):
	print 500.0*np.round(np.reshape([a_state_action[0][0:-4]],(21,12)),3)+100.0
	# print np.reshape(a_state_action[0:252:1],(21,12))
def nn_to_human(a_state_action): return np.clip(50.0*10.0*np.reshape([a_state_action[0][0:-4]],(21,12))+100.0,0,255)
nn_view = 128.0*np.ones((21,12))
nn_mem_view = 128.0*np.ones((21,12))

actions =  [ [[20,0,0,0]] , [[0,20,0,0]] , [[0,0,20,0]] , [[0,0,0,20]] ]
n_actions = len(actions)
q_values_1 = [0] * n_actions
q_values_2 = [0] * n_actions
state_actions_1 = [0] * n_actions
state_actions_2 = [0] * n_actions
reward_2 = 0.0

# play and learn

print 'playing..'

for time in xrange(10000000):

	# 1st animation call of 2

	if show: animate()

	# set reward to zero
	
	reward_2 = 0.0
	# reward_2 += 0.000001 #= 0.0

	# get q values

	state_1 = get_state()
	for i in xrange(n_actions): 
		state_actions_1[i] = get_state_action(state_1,actions[i])
		q_values_1[i] = test(state_actions_1[i])[0][0][0]

	# select move with largest q value

	move = np.argmax(q_values_1)
	move_accounting[move] += 1
	if show: nn_view = nn_to_human(state_actions_1[move])

	# play the move

	if   move == 0: move_p_left()
	elif move == 1: move_p_right()
	elif move == 2: rotate_p()
	elif move == 3: pass # do nothing

	# 2nd animation call of 2

	if show: animate()

	# apply gravity

	move_p_down()

	# detect game over and reset board

	terminal_2 = False
	if collision(): # it's weird that the nn doesn't see this
		# print 'game over'
		# reward_2 -= 0.05 # discourage
		reward_2 = 0.0
		terminal_2 = True
		game_count += 1
		if show: animate_game_over()
		clear_board()
		p = g.next()
		next_p = g.next()
		p_x = 7
		p_y = 0
		add_p(p_x,p_y)

	# get subsequent state

	state_2 = get_state()

	# store transition memory

	memory[time%mem_size] = [state_actions_1[move], reward_2, state_2, terminal_2]
	priority[time%mem_size] = max_priority

	# learn from memories

	if time > init_watch_time: # and time % (watch_time+1) == 0:

		# for reflection in xrange(min(time+1,2)):
		# for memory_index in [rng.randint((time+1)%mem_size),np.argmax(priority)]:
		# for memory_index in [rng.randint((time+1)%mem_size),rng.randint((time+1)%mem_size)]:
		# important_memory = np.argmax(priority)
		# nn_mem_view = nn_to_human(memory[important_memory][0])

		for reflection in xrange(4):

			if reflection<2: 
				memory_index = np.argmax(priority)
			else:
				memory_index = rng.randint(time%mem_size+1)

			# np.argmax(priority),rng.randint(time%mem_size+1):
			# pick a random memory 
			# memory_index = rng.randint((time+1)%mem_size)
			# pick the highest priority transition memory
			# memory_index = np.argmax(priority)
			# pick a transition memory with P ~ priority
			# memory_index = np.argmax(np.cumsum(priority)>rng.uniform(np.sum(priority)))

			# recall

			state_action_1, reward_2, state_2, terminal_2 = memory[memory_index]
			if reflection==1: nn_mem_view = nn_to_human(state_action_1)

			# get target

			if terminal_2:
				y_2 = reward_2
			else:
				for i in xrange(n_actions):
					state_actions_2[i] = get_state_action(state_2,actions[i])
					q_values_2[i] = test(state_actions_2[i])[0][0][0]
				# y_2 = reward_2 + gamma * np.max(q_values_2) # max here results in overestimates resolved by double q learning
				y_2 = reward_2 + gamma * target_test(state_actions_2[np.argmax(q_values_2)])[0][0][0]

			# learn

			y_2_guess, error = train(state_action_1,[[y_2]])
			priority[memory_index] = error

	# periodically update target network

	if(time%100==0): target_update()

	# logging

	if time%1000==0: # or time<10:
		print "time =",time,
		print "line clears =",line_clears,
		print "games =",game_count,
		# print "cost =",cost
		print "moves =",move_accounting,
		# print "q's =",q_values,
		# print "q test =", np.max(q_values),
		# print "q train =",y_2_guess[0][0],
		# print "error",error,
		print "epsilon =",epsilon,
		print "gamma =",gamma,
		print "r =",reward_2,
		print "learning rate =",learning_rate
		print "b["+str(m)+"]", b[m].get_value()[0],
		print "W["+str(m)+"][0]", W[m].get_value()[0]
		# print "Wb", np.mean(np.mean([W[i].get_value() for i in xrange(m+1)])),
		# print "bb", np.mean(np.mean([b[i].get_value() for i in xrange(m+1)])),
		for i in xrange(m+1): print "W["+str(i)+"] =",np.mean(W[i].get_value()),np.sum(W[i].get_value()),
		print ''
		for i in xrange(m+1): print "b["+str(i)+"] =",np.mean(b[i].get_value()),np.sum(b[i].get_value()),
		print ''
		# print "error =",error
		# pretty_print_board()
		# learning_rate -= 0.01 * (learning_rate - 0.001)
		# gamma += 0.05 * (0.98 - gamma)
		# epsilon -= 0.05 * (epsilon - 0.01)
		# print np.max(priority)
		# print memory[np.argmax(priority)][2]
		if time>init_watch_time:
			print "y =",y_2,
			print "y_guess =",y_2_guess[0][0],
			print "error =",np.max(priority)
			if np.isnan(error): 
				print 'OD'
				sys.exit()
		# print_state(memory[np.argmax(priority)][2]) # need to unfocus
		# nn_view = nn_to_human(memory[np.argmax(priority)][0])
		pretty_print_board()
		pretty_print_state(memory[np.argmax(priority)][2])
		print "reward =", memory[np.argmax(priority)][1],
		print "terminal =", memory[np.argmax(priority)][3]


		
	# save model

	if (time%10001==10000): # and (len(sys.argv) > 3) and (sys.argv[3] == 'save'):
		# filename='model'+str(line_clears)+'in'+str(game_count)+'using'+str(m)+'layers'+'.save'
		filename='model.save'
		print '-- saving model as',filename
		save_file = open(filename, 'wb')
		for param in params:
			cPickle.dump(param, save_file, -1)
		save_file.close()

	if (time%100001==100000): # and (len(sys.argv) > 3) and (sys.argv[3] == 'save'):
		filename='model'+str(line_clears)+'in'+str(game_count)+'using'+str(m)+'layers'+'.save'
		# filename='model.save'
		print '-- saving model as',filename
		save_file = open(filename, 'wb')
		for param in params:
			cPickle.dump(param, save_file, -1)
		save_file.close()

print 'finished playing at',datetime.datetime.now()

# notes and todo 

# chase after high error states!
# ie estimate ambiguity/confidence

# purge nostalgia

# have nn estimate novelty, test with memory?

# inject noise

# 2 memories, 1 random, 1 prioritized

# computer memory

# hash functions, hash for novelty


# smaller view window would dramatically improve results
# learn what the ideal looks like
# have a theory as to what the ideal looks like

# attention is ai-complete?
# attention is ai? sort of?
# try to predict input
# separate NNs for actions and states and Q estimation integrated
# hack the matrix curriculum
# normalize colors..
# terminal is a form of cheating
# r shouldn't overwhelm q value...
# actually terminate the game...
# organize hyperparameters.. and notes..
# actions should be chosen based on novelty, not just r
# incorporate novelty into r
# experience replay
# convolutions
# normalize color
# just state instead of state-action
# overstimulation -- neurons that fired last turn can't fire this one
# convolution simulates attention and symmetry!!! humans use cnns!!! humans appreciate symmetry!!!
# aiming a p just like driving a car
# p_x is too much rolling cause board is bigger
# test game engine and train from human play
# record which ps it succeeds the most with, more accounting etc
# visualize learning the weights
# actually reset
# dqn guesses upper and lower bounds, multiple outputs etc, chooses action w least upper bound
# separate nn to process actions
# disentangle / establish independences


# if output > 0 -> recurse/loop
# neuralize rs
# end-to-end RL

# grid lstm

# g_u = s(W_u*H)
# g_f = s(W_f*H)
# g_o = s(W_o*H)
# g_c = tanh(W_c*H)
# m_p = g_f & m + g_u & g_c
# h_p = tanh(g_o & m_p)

# h_p, m_p = LSTM(H,m,W)

# lstm

# f_t =    s(dot([h_tm1,x_t],W_f)+b_f)
# i_t =    s(dot([h_tm1,x_t],W_i)+b_i)
# c_t = tanh(dot([h_tm1,x_t],W_C)+b_C)
# C_t = f_t * C_tm1 + i_t * c_t
# o_t =    s(dot([h_tm1,x_t],W_o)+b_o)
# h_t = o_t * tanh(C_t)