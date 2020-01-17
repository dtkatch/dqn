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
# import pygame
# from pygame.locals import *



# tetris logic

kinds = [
		[[0,11,0],[11,11,11],[0,0,0]],
		[[0,0,12],[12,12,12],[0,0,0]],
		[[0,0,0,0],[13,13,13,13],[0,0,0,0],[0,0,0,0]],
		[[14,14],[14,14]],
		[[15,15,0],[0,15,15],[0,0,0]],
		[[16,0,0],[16,16,16],[0,0,0]],
		[[0,17,17],[17,17,0],[0,0,0]]
		]

seq = numpy.random.permutation(7)

flag_4x1 = False
def piece_generator():
	while True:
		# seq = numpy.random.permutation(7) # normal tetris
		if flag_4x1:
			seq = numpy.array([2,2,2,2,2,2,2]) # lines
		else:
			seq = numpy.array([3,3,3,3,3,3,3]) # blocks
		# seq = numpy.array([5,5,5,5,5,5,5]) # Ls
		# seq = numpy.random.randint(2,4) * np.array([1,1,1,1,1,1,1])
		for i in range(7):
			yield kinds[seq[i]]

g = piece_generator()

p = g.next()
next_p = g.next()

p_x = 6
p_y = 0

length = 20
width = 10

board = [[0,0,10]+[0 for x in range(width)]+[10,0,0] for y in range(length)] \
		+[[0,0]+[10 for x in range(width+2)]+[0,0]] \
		+[[0 for x in range(width+6)]] \
		+[[0 for x in range(width+6)]]

def rotate_array(a):
	return numpy.array(zip(*a[::-1]))

def add_p(x,y):
	for i,j in itertools.product(xrange(len(p)),xrange(len(p[0]))):
		board[y+j][x+i] += p[i][j]

def remove_p(x,y):
	for i,j in itertools.product(xrange(len(p)),xrange(len(p[0]))):
		board[y+j][x+i] -= p[i][j]

def collision():
	for i in xrange(length+3):
		for j in xrange(width+6):
			if board[i][j] > 20:
				return True
	return False

def rotate_p():
	global p, p_x, p_y
	remove_p(p_x,p_y)
	p = rotate_array(p)
	add_p(p_x,p_y)

	if collision():

		# wall kick
		move_p_left()
		if collision():
			move_p_right()
			if collision():
				move_p_left()
			else:
				return
		else:
			return

		remove_p(p_x,p_y)
		p = rotate_array(p)
		p = rotate_array(p)
		p = rotate_array(p)
		add_p(p_x,p_y)

def clear_filled_rows():
	for i in range(length-1,-1,-1):
		full = True
		for j in range(3,width+3,1):
			if board[i][j] == 0:
				full = False
		if full:
			global r
			r = r + 0.05 # reward
			print "line cleared"
			if show:
				draw_board()
				flash_green()
				pygame.display.update()
			for j in range(3,width+3,1):
				board[i][j] = 0
			for k in range(i,0,-1):
				for j in range(3,width+3,1):
					board[k][j] = board[k-1][j]
			clear_filled_rows()

def clear_board():
	for i in range(length-1,-1,-1):
		for j in range(3,width+3,1):
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

		# spawn new piece
		p_x = 6
		p_y = 0
		p = next_p
		next_p = g.next()
		add_p(p_x,p_y)

		# tell drop_p() to stop
		return False

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

add_p(p_x,p_y)

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
	else: 
		return PINK

def draw_board():
	global board, length, width
	for i in xrange(length+3):
		for j in xrange(width+6):
			pygame.draw.rect(windowSurface, color_map(board[i][j]), (100+j*20, 50+i*20, 19, 19))
	for i in xrange(len(p)):
		for j in xrange(len(p[0])):
			pygame.draw.rect(windowSurface, BLACK, (450+j*20, 100+i*20, 19, 19))
	for i in xrange(len(next_p)):
		for j in xrange(len(next_p[0])):
			pygame.draw.rect(windowSurface, color_map(next_p[i][j]), (450+j*20, 100+i*20, 19, 19))

def flash_red():
	for i in xrange(length+3):
		for j in xrange(width+6):
			if board[i][j] != 0:
				pygame.draw.rect(windowSurface, RED, (100+j*20, 50+i*20, 19, 19))	

def flash_green():
	for i in xrange(length+3):
		for j in xrange(width+6):
			if board[i][j] != 0:
				pygame.draw.rect(windowSurface, GREEN, (100+j*20, 50+i*20, 19, 19))	

# dqn

actions =  [ [[1,0,0,0]] , [[0,1,0,0]] , [[0,0,1,0]] , [[0,0,0,1]] ]
# actions =  [ [[1,0,0,0,0,0]] , [[0,1,0,0,0,0]] , [[0,0,1,0,0,0]] , [[0,0,0,1,0,0]] , [[0,0,0,0,1,0]] , [[0,0,0,0,0,1]] ]

q = [0,0,0,0]
qq = [0,0,0,0]
# q = [0,0,0,0,0,0]
# qq = [0,0,0,0,0,0]

# def sig(x,deriv=False): return x*(1-x) if deriv else 1/(1+np.exp(-x))
def sig(x,deriv=False): return x*(1-x) if deriv else 2/(1+np.exp(-x))-1
np.random.seed(0)

# init/load model

load = False
if len(sys.argv) > 1 and sys.argv[1] == 'load':
	load = True

if load:
	filename='model.save'
	print 'model', filename, 'selected'
	print 'loading model..'
	save_file = open(filename, 'rb') 
	W0 = cPickle.load(save_file)
	W1 = cPickle.load(save_file)
	W2 = cPickle.load(save_file)
	W3 = cPickle.load(save_file)
	save_file.close()
else:
	print 'no model selected'
	print 'initializing new model..' # no biases, no normalization, arbitrary scaling
	W0 = ( 2*np.random.random((200+4,200+4)) - 1 ) * 0.5
	W1 = ( 2*np.random.random((200+4,100)) - 1 ) * 0.5
	W2 = ( 2*np.random.random((100,50)) - 1 ) * 0.5
	W3 = ( 2*np.random.random((50,1)) - 1 ) * 0.5 * 0.01

# q learning params

gamma = 0.9 # 0.995 # 0.99
epsilon = 0.001 # 0.05

# show games (100x slower, 1.5x more entertaining)

show = False
if len(sys.argv) > 2 and sys.argv[2] == 'show':
	show = True

if show:
	import pygame
	from pygame.locals import *
	pygame.init()
	windowSurface = pygame.display.set_mode((800, 600), 0, 32)
	windowSurface.fill(BLACK)

# play and kinda learn

for j in range(10000):

	# set reward to zero

	r = 0

	# pieces fall in tetris

	move_p_down()

	# detect game over and reset board

	if collision():
		r = r - 0.0001 # punish
		print 'game over'
		if show:
			draw_board()
			flash_red()
			pygame.display.update()
			# pygame.time.wait(500)
		if j>5000 and flag_4x1==False: 
			flag_4x1=True
			for i in range(7): g.next()
			print '5000 moves played'
			print 'introducing new piece'
		clear_board()
		p = g.next()
		next_p = g.next()
		p_x = 6
		p_y = 0
		add_p(p_x,p_y)
		if show: draw_board()

	# figure out best move/action

	s = 0.05 * np.reshape([subl[3:width+3] for subl in board[0:length]],(1,200))
	for i in range(len(actions)):
		a = actions[i]
		X = np.concatenate((s,a), axis=1)
		X = X * 0.1
		l0 = X
		l1 = sig(np.dot(l0,W0))
		l2 = sig(np.dot(l1,W1))
		l3 = sig(np.dot(l2,W2))
		l4 = sig(np.dot(l3,W3)) 
		q[i] = l4[0][0]

	move = np.argmax(q) if np.random.random() > epsilon else np.random.randint(len(q))

	# play best move

	if move == 0: move_p_left()
	if move == 1: move_p_right()
	if move == 2: rotate_p()
	if move == 3: drop_p()
	# if move == 4:
	# 	rotate_p()
	# 	rotate_p()
	# 	rotate_p()
	# if move == 5: pass
	# if move == 6: down_p()

	# show game

	if show:
		draw_board()
		pygame.display.update()
		pygame.event.clear()

	# estimate q value of subsequent state

	ss = 0.05 * np.reshape([subl[3:width+3] for subl in board[0:length]],(1,200))
	for i in range(len(actions)):
		aa = actions[i]
		X = np.concatenate((ss,aa), axis=1)
		X = X * 0.1
		ll0 = X # 4 3, 368 
		ll1 = sig(np.dot(ll0,W0)) # 4 4
		ll2 = sig(np.dot(ll1,W1)) # 4 2
		ll3 = sig(np.dot(ll2,W2)) # 4 1
		ll4 = sig(np.dot(ll3,W3)) 
		qq[i] = ll4[0][0]

	# compute target value

	y = r + gamma * np.max(qq)

	# unnecessarily recompute q value for best action

	a = actions[move]
	X = np.concatenate((s,a), axis=1)
	# print X
	X = X * 0.1
	l0 = X # 4 3, 368 
	l1 = sig(np.dot(l0,W0)) # 4 4
	l2 = sig(np.dot(l1,W1)) # 4 2
	l3 = sig(np.dot(l2,W2)) # 4 1
	l4 = sig(np.dot(l3,W3)) 

	# compute error

	l4_err = y - l4
	l4_err = l4_err * 0.5

	# logging

	# print l4_err
	# if j%100==0: 
		# print "j",j,"move", move,"l4", l4, "y", y, "l4_err", l4_err, "W", W3[25][0], "q", q
	# if r != 0: print r, qq
	# if np.mean(np.abs(l4_err)) > 0.1: print l4_err

	# backprop

	l4_del = l4_err				 * sig(l4,True)
	l3_del = np.dot(l4_del,W3.T) * sig(l3,True)
	l2_del = np.dot(l3_del,W2.T) * sig(l2,True)
	l1_del = np.dot(l2_del,W1.T) * sig(l1,True)


	# weight update

	W0 += np.dot(l0.T,l1_del)
	W1 += np.dot(l1.T,l2_del)
	W2 += np.dot(l2.T,l3_del) 
	W3 += np.dot(l3.T,l4_del)

# save model

save = False
if len(sys.argv) > 3 and sys.argv[3] == 'save':
	save = True

if save:
	print 'saving model..'
	params = W0, W1, W2, W3
	def save_model(filename='model.save'):
	    save_file = open(filename, 'wb') 
	    for p in params:
	        cPickle.dump(p, save_file, -1)
	    save_file.close()
	save_model()

# quit

if show:
	pygame.quit()
	sys.exit()
