#!/usr/bin/python3

"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
Slightly modified by Nthend
"""

import numpy as np


class Session:
	def __init__(self, datafn):
		# data I/O
		self.datafn = datafn
		self.data = open(self.datafn, 'r', encoding='utf-8').read()  # should be simple plain text file

		# hyperparameters
		self.hidden_size = 100  # size of hidden layer of neurons
		self.seq_length = 25  # number of steps to unroll the RNN for
		self.learning_rate = 1e-1

		self.new()

	def new(self):
		self.chars = sorted(list(set(self.data)))
		self.Wxh = np.random.randn(self.hidden_size, self.vocab_size)*0.01  # input to hidden
		self.Whh = np.random.randn(self.hidden_size, self.hidden_size)*0.01  # hidden to hidden
		self.Why = np.random.randn(self.vocab_size, self.hidden_size)*0.01  # hidden to output
		self.bh = np.zeros((self.hidden_size, 1))  # hidden bias
		self.by = np.zeros((self.vocab_size, 1))  # output bias

	def load(self, fname):
		dmp = np.load(fname)
		self.chars = list(dmp['chars'])
		self.Wxh = dmp['Wxh']  # input to hidden
		self.Whh = dmp['Whh']  # hidden to hidden
		self.Why = dmp['Why']  # hidden to output
		self.bh = dmp['bh']  # hidden bias
		self.by = dmp['by']  # output bias

	def update(self):
		self.data_size, self.vocab_size = len(self.data), len(self.chars)
		print('data has %d characters, %d unique.' % (data_size, vocab_size))
		print(chars)
		char_to_ix = {ch: i for i, ch in enumerate(chars)}
		ix_to_char = {i: ch for i, ch in enumerate(chars)}


def lossFun(inputs, targets, hprev):
	"""
	inputs,targets are both list of integers.
	hprev is Hx1 array of initial hidden state
	returns the loss, gradients on model parameters, and last hidden state
	"""
	xs, hs, ys, ps = {}, {}, {}, {}
	hs[-1] = np.copy(hprev)
	loss = 0
	# forward pass
	for t in range(len(inputs)):
		xs[t] = np.zeros((vocab_size, 1))  # encode in 1-of-k representation
		xs[t][inputs[t]] = 1
		hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh)  # hidden state
		ys[t] = np.dot(Why, hs[t]) + by  # unnormalized log probabilities for next chars
		ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))  # probabilities for next chars
		loss += -np.log(ps[t][targets[t], 0])  # softmax (cross-entropy loss)
	# backward pass: compute gradients going backwards
	dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
	dbh, dby = np.zeros_like(bh), np.zeros_like(by)
	dhnext = np.zeros_like(hs[0])
	for t in reversed(range(len(inputs))):
		dy = np.copy(ps[t])
		dy[targets[t]] -= 1  # backprop into y
		dWhy += np.dot(dy, hs[t].T)
		dby += dy
		dh = np.dot(Why.T, dy) + dhnext  # backprop into h
		dhraw = (1 - hs[t] * hs[t]) * dh  # backprop through tanh nonlinearity
		dbh += dhraw
		dWxh += np.dot(dhraw, xs[t].T)
		dWhh += np.dot(dhraw, hs[t-1].T)
		dhnext = np.dot(Whh.T, dhraw)
	for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
		np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients
	return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]


def sample(h, seed_ix, n):
	"""
	sample a sequence of integers from the model
	h is memory state, seed_ix is seed letter for first time step
	"""
	x = np.zeros((vocab_size, 1))
	x[seed_ix] = 1
	ixes = []
	for t in range(n):
		h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
		y = np.dot(Why, h) + by
		p = np.exp(y) / np.sum(np.exp(y))
		ix = np.random.choice(range(vocab_size), p=p.ravel())
		x = np.zeros((vocab_size, 1))
		x[ix] = 1
		ixes.append(ix)
	return ixes

if load:
	hprev = dmp['hprev']
	n, p = dmp['n'], dmp['p']
	mWxh, mWhh, mWhy = dmp['mWxh'], dmp['mWhh'], dmp['mWhy']
	mbh, mby = dmp['mbh'], dmp['mby']  # memory variables for Adagrad
else:
	n, p = 0, 0
	mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
	mbh, mby = np.zeros_like(bh), np.zeros_like(by)  # memory variables for Adagrad

smooth_loss = -np.log(1.0/vocab_size)*seq_length  # loss at iteration 0


done = False

plot = True
passes = []
losses = []

while not done:
	if n % 100 == 0:
		pass  # display.clear_output(wait=True)

	# prepare inputs (we're sweeping from left to right in steps seq_length long)
	if p+seq_length+1 >= len(data) or n == 0:
		hprev = np.zeros((hidden_size, 1))  # reset RNN memory
		p = 0  # go from start of data
	inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
	targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

	# forward seq_length characters through the net and fetch gradient
	loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
	smooth_loss = smooth_loss * 0.999 + loss * 0.001
	if n % 100 == 0:
		book_pass = n*seq_length/len(data)
		if plot:
			passes.append(book_pass)
			losses.append(smooth_loss)
			# ax = plt.figure().add_subplot(111)
			# ax.plot(passes, losses)
			# plt.show()
		else:
			print('pass: %f, loss: %f' % (book_pass, smooth_loss))  # print progress

	# perform parameter update with Adagrad
	for param, dparam, mem in zip(
		[Wxh, Whh, Why, bh, by],
		[dWxh, dWhh, dWhy, dbh, dby],
		[mWxh, mWhh, mWhy, mbh, mby]
	):
		mem += dparam * dparam
		param += -learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update

	p += seq_length  # move data pointer
	n += 1  # iteration counter


sample_ix = sample(hprev, inputs[0], 4000)
txt = ''.join(ix_to_char[ix] for ix in sample_ix)
print(txt)


def save(fname):
	dmp = {
		# char map
		'chars': np.array(chars),
		# counters
		'n': n,
		'p': p,
		# rnn memory
		'hprev': hprev,
		# weights and biases
		'Wxh': Wxh,
		'Whh': Whh,
		'Why': Why,
		'bh': bh,
		'by': by,
		# adagrad vars
		'mWxh': mWxh,
		'mWhh': mWhh,
		'mWhy': mWhy,
		'mbh': mbh,
		'mby': mby
	}
	np.savez_compressed(fname, **dmp)


# PyQt4

import sys
from PyQt4 import QtGui


app = QtGui.QApplication(sys.argv)


class MainWindow(QtGui.QMainWindow):
	def __init__(self):
		super(MainWindow, self).__init__()

		# menu bar
		self.menubar = self.menuBar()
		mfile = self.menubar.addMenu('&File')
		aexit = QtGui.QAction('&Exit', self)
		aexit.setShortcut('Ctrl+Q')
		aexit.triggered.connect(self.close)
		mfile.addAction(aexit)

		# status bar
		self.statusBar().showMessage('Ready')

		# quit button
		self.bquit = QtGui.QPushButton('Quit', self)
		self.bquit.move(0, 100)
		self.bquit.clicked.connect(self.close)

		# construct window
		self.resize(800, 600)
		self.setWindowTitle('Simple')

	def closeEvent(self, event):
		reply = QtGui.QMessageBox.question(self, 'Message', "Are you sure to quit?", QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.No)
		if reply == QtGui.QMessageBox.Yes:
			event.accept()
		else:
			event.ignore()

w = MainWindow()
w.show()

sys.exit(app.exec_())
