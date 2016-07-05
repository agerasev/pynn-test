#!/usr/bin/python3

"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
Slightly modified by Nthend
"""

import sys
from threading import Thread

import numpy as np

from PyQt4 import QtGui


class Session:
	def __init__(self, datafn):
		# data I/O
		self.datafn = datafn
		self.data = open(self.datafn, 'r', encoding='utf-8').read()  # should be simple plain text file
		self.data_size = len(self.data)

		self.chars = sorted(list(set(self.data)))
		self.vocab_size = len(self.chars)

		# hyperparameters
		self.hidden_size = 100  # size of hidden layer of neurons
		self.seq_length = 25  # number of steps to unroll the RNN for
		self.learning_rate = 1e-1

		self.new()

	def new(self):
		self.Wxh = np.random.randn(self.hidden_size, self.vocab_size)*0.01  # input to hidden
		self.Whh = np.random.randn(self.hidden_size, self.hidden_size)*0.01  # hidden to hidden
		self.Why = np.random.randn(self.vocab_size, self.hidden_size)*0.01  # hidden to output
		self.bh = np.zeros((self.hidden_size, 1))  # hidden bias
		self.by = np.zeros((self.vocab_size, 1))  # output bias

		self.n, self.p = 0, 0
		self.mWxh, self.mWhh, self.mWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
		self.mbh, self.mby = np.zeros_like(self.bh), np.zeros_like(self.by)  # memory variables for Adagrad

		self.smooth_loss = -np.log(1.0/self.vocab_size)*self.seq_length  # loss at iteration 0

		self.update()

	def load(self, fname):
		dmp = np.load(fname)

		self.Wxh = dmp['Wxh']  # input to hidden
		self.Whh = dmp['Whh']  # hidden to hidden
		self.Why = dmp['Why']  # hidden to output
		self.bh = dmp['bh']  # hidden bias
		self.by = dmp['by']  # output bias

		self.hprev = dmp['hprev']
		self.n, self.p = dmp['n'], dmp['p']
		self.mWxh, self.mWhh, self.mWhy = dmp['mWxh'], dmp['mWhh'], dmp['mWhy']
		self.mbh, self.mby = dmp['mbh'], dmp['mby']  # memory variables for Adagrad

		self.smooth_loss = dmp['smooth_loss']

		self.update()

	def update(self):
		print('data has %d characters, %d unique.' % (self.data_size, self.vocab_size))
		print(self.chars)
		self.char_to_ix = {ch: i for i, ch in enumerate(self.chars)}
		self.ix_to_char = {i: ch for i, ch in enumerate(self.chars)}

	def lossFun(self, inputs, targets, hprev):
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
			xs[t] = np.zeros((self.vocab_size, 1))  # encode in 1-of-k representation
			xs[t][inputs[t]] = 1
			hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh)  # hidden state
			ys[t] = np.dot(self.Why, hs[t]) + self.by  # unnormalized log probabilities for next chars
			ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))  # probabilities for next chars
			loss += -np.log(ps[t][targets[t], 0])  # softmax (cross-entropy loss)
		# backward pass: compute gradients going backwards
		dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
		dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
		dhnext = np.zeros_like(hs[0])
		for t in reversed(range(len(inputs))):
			dy = np.copy(ps[t])
			dy[targets[t]] -= 1  # backprop into y
			dWhy += np.dot(dy, hs[t].T)
			dby += dy
			dh = np.dot(self.Why.T, dy) + dhnext  # backprop into h
			dhraw = (1 - hs[t] * hs[t]) * dh  # backprop through tanh nonlinearity
			dbh += dhraw
			dWxh += np.dot(dhraw, xs[t].T)
			dWhh += np.dot(dhraw, hs[t-1].T)
			dhnext = np.dot(self.Whh.T, dhraw)
		for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
			np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients
		return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

	def sample(self, h, seed_ix, n):
		"""
		sample a sequence of integers from the model
		h is memory state, seed_ix is seed letter for first time step
		"""
		x = np.zeros((self.vocab_size, 1))
		x[seed_ix] = 1
		ixes = []
		for t in range(n):
			h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
			y = np.dot(self.Why, h) + self.by
			p = np.exp(y) / np.sum(np.exp(y))
			ix = np.random.choice(range(self.vocab_size), p=p.ravel())
			x = np.zeros((self.vocab_size, 1))
			x[ix] = 1
			ixes.append(ix)
		return ixes

	def learn(self):
		# prepare inputs (we're sweeping from left to right in steps seq_length long)
		if self.p + self.seq_length + 1 >= len(self.data) or self.n == 0:
			self.hprev = np.zeros((self.hidden_size, 1))  # reset RNN memory
			self.p = 0  # go from start of data
		inputs = [self.char_to_ix[ch] for ch in self.data[self.p:self.p+self.seq_length]]
		targets = [self.char_to_ix[ch] for ch in self.data[self.p+1:self.p+self.seq_length+1]]

		# forward seq_length characters through the net and fetch gradient
		loss, dWxh, dWhh, dWhy, dbh, dby, self.hprev = self.lossFun(inputs, targets, self.hprev)
		self.smooth_loss = self.smooth_loss * 0.999 + loss * 0.001

		# perform parameter update with Adagrad
		for param, dparam, mem in zip(
			[self.Wxh, self.Whh, self.Why, self.bh, self.by],
			[dWxh, dWhh, dWhy, dbh, dby],
			[self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby]
		):
			mem += dparam * dparam
			param += -self.learning_rate*dparam/np.sqrt(mem + 1e-8)  # adagrad update

		self.p += self.seq_length  # move data pointer
		self.n += 1  # iteration counter

	def generate(self, length):
		sample_ix = self.sample(self.hprev, self.char_to_ix[self.data[self.p]], length)
		txt = ''.join(self.ix_to_char[ix] for ix in sample_ix)
		return txt

	def passes(self):
		return self.n*self.seq_length/len(self.data)

	def save(self, fname):
		dmp = {
			# counters
			'n': self.n,
			'p': self.p,
			# rnn memory
			'hprev': self.hprev,
			# weights and biases
			'Wxh': self.Wxh,
			'Whh': self.Whh,
			'Why': self.Why,
			'bh': self.bh,
			'by': self.by,
			# adagrad vars
			'mWxh': self.mWxh,
			'mWhh': self.mWhh,
			'mWhy': self.mWhy,
			'mbh': self.mbh,
			'mby': self.mby
		}
		np.savez_compressed(fname, **dmp)


class MainWindow(QtGui.QWidget):
	class BookWidget(QtGui.QWidget):
		def __init__(self, outer):
			self.outer = outer
			QtGui.QWidget.__init__(self)

			hbox = QtGui.QHBoxLayout()
			bload = QtGui.QPushButton('Load Book')
			# bload.clicked.connect()
			hbox.addStretch(1)
			hbox.addWidget(bload)

			self.setLayout(hbox)

	class SessionWidget(QtGui.QWidget):
		def __init__(self, outer):
			self.outer = outer
			QtGui.QWidget.__init__(self)

			hbox = QtGui.QHBoxLayout()

			hbox.addStretch(1)

			bload = QtGui.QPushButton('Load')
			# bload.clicked.connect()
			hbox.addWidget(bload)

			bsave = QtGui.QPushButton('Save')
			# bsave.clicked.connect()
			hbox.addWidget(bsave)

			self.setLayout(hbox)

	def __init__(self):
		QtGui.QWidget.__init__(self)

		"""
		# menu bar
		self.menubar = self.menuBar()
		mfile = self.menubar.addMenu('&File')
		aexit = QtGui.QAction('&Exit', self)
		aexit.setShortcut('Ctrl+Q')
		aexit.triggered.connect(self.close)
		mfile.addAction(aexit)

		# status bar
		self.statusBar().showMessage('Ready')
		"""

		# elements
		vbox = QtGui.QVBoxLayout()

		vbox.addWidget(self.BookWidget(self))
		vbox.addWidget(self.SessionWidget(self))
		vbox.addStretch(1)

		self.setLayout(vbox)

		# construct window
		self.resize(800, 600)
		self.setWindowTitle('Simple')

		# init session
		self.session = Session('data/witcher_rus.txt')

		# learn thread
		self.lthread = Thread(target=self.learn, args=())

	def learn(self):
		while not self.done:
			self.session.learn()
			if self.session.n % 100 == 0:
				print(self.session.smooth_loss)
				print(self.session.passes())
				print(self.session.generate(1000))

	def startSession(self):
		self.done = False
		self.lthread.start()

	def stopSession(self):
		self.done = True
		self.lthread.join()

	def closeEvent(self, event):
		reply = QtGui.QMessageBox.question(
			self, 'Save progress', "Do you want to save learned network?",
			QtGui.QMessageBox.Save | QtGui.QMessageBox.Discard | QtGui.QMessageBox.Cancel, QtGui.QMessageBox.Cancel
		)
		if reply == QtGui.QMessageBox.Save or reply == QtGui.QMessageBox.Discard:
			self.stopSession()
			event.accept()
		else:
			event.ignore()


app = QtGui.QApplication(sys.argv)

w = MainWindow()
w.show()
w.startSession()

sys.exit(app.exec_())
