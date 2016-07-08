#!/usr/bin/python3

import numpy as np

import pynn as nn


data = '''
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Morbi ut purus nec dolor aliquam egestas.
Quisque pharetra sed orci ut sollicitudin. Nulla facilisi. Vestibulum at consequat ipsum.
Mauris eu luctus elit, vitae sagittis nisl. Phasellus ex eros, dignissim in vehicula sit amet, dignissim id dui.
Vestibulum non lacus auctor, cursus turpis a, posuere ipsum. Nunc convallis dictum lacus quis egestas.
Vivamus quis urna justo. Suspendisse sed sapien vitae sapien placerat ultricies.
Proin elementum odio id ante dictum, a porttitor orci efficitur.
Aenean elementum nulla nibh, id venenatis magna tristique in. Donec lacinia semper faucibus.
In vitae porta nunc, in tempor est. Nam vel massa ultricies, aliquet ligula suscipit, suscipit odio.
Vivamus consequat porta ligula ut interdum.
'''

chars = sorted(list(set(data)))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

# hyperparameters
hidden_size = 100  # size of hidden layer of neurons
seq_length = 25  # number of steps to unroll the RNN for
learning_rate = 1e-1

# model parameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.01  # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01  # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01  # hidden to output
bh = np.zeros((hidden_size, 1))  # hidden bias
by = np.zeros((vocab_size, 1))  # output bias

istate = np.zeros((hidden_size, 1))

opt = {
	'clip': 5,
	'adagrad': True,
	'adagrad.eps': 1e-8,
	'rate': learning_rate
}

net = nn.Network(
	[
		nn.Matrix((vocab_size, hidden_size), weight=Wxh.T, **opt),

		nn.Fork(**opt),
		nn.Bias(hidden_size, weight=bh.reshape(hidden_size), **opt),
		nn.Tanh(**opt),
		nn.Fork(**opt),

		nn.Depot(weight=istate.reshape(hidden_size), **opt),
		nn.Matrix((hidden_size, hidden_size), weight=Whh.T, **opt),

		nn.Matrix((hidden_size, vocab_size), weight=Why.T, **opt),
		nn.Bias(vocab_size, weight=by.reshape(vocab_size), **opt),
		nn.SoftmaxLoss(**opt)
	],
	[
		((-1, 1), (0, 1)),

		((0, 2), (1, 2)),

		((1, 1), (2, 1)),
		((2, 2), (3, 1)),
		((3, 2), (4, 1)),
		((4, 2), (5, 1)),
		((5, 2), (6, 1)),
		((6, 2), (1, 3)),

		((4, 3), (7, 1)),

		((7, 2), (8, 1)),
		((8, 2), (9, 1)),
		((9, 2), (-1, 2))
	]
)

net.push(0, 'emit')


class Handler:
	def __init__(self):
		self.out = None

	def __call__(self, ch, sig):
		if ch == 2:
			self.out = sig


net.emit = Handler()


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

		(a, b) = (hs[t-1].reshape(hidden_size), net.nodes[5].slot)
		if not np.allclose(a, b):
			print('diff: %f' % np.sum((a - b)**2))
			raise Exception('hidden state mismatch')

		net.push(1, xs[t].reshape(vocab_size))
		net.push(0, 'release')

		(a, b) = (ps[t].reshape(vocab_size), net.emit.out)
		if not np.allclose(a, b):
			print('diff: %f' % np.sum((a - b)**2))
			raise Exception('forward output mismatch')

	print('[ ok ] forward test passed')

	# backward pass: compute gradients going backwards
	dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
	dbh, dby = np.zeros_like(bh), np.zeros_like(by)
	dhnext = np.zeros_like(hs[0])

	net.push(0, 'emit.error')

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

		net.push(2, targets[t])

	(a, b) = (dWhy.T, net.nodes[7].grad)
	if not np.allclose(a, b):
		print('diff: %f' % np.sum((a - b)**2))
		raise Exception('backward Why grad mismatch')

	(a, b) = (dWhh.T, net.nodes[6].grad)
	if not np.allclose(a, b):
		print('diff: %f' % np.sum((a - b)**2))
		raise Exception('backward Whh grad mismatch')

	(a, b) = (dWxh.T, net.nodes[0].grad)
	if not np.allclose(a, b):
		print('diff: %f' % np.sum((a - b)**2))
		raise Exception('backward Wxh grad mismatch')

	(a, b) = (hs[len(inputs) - 1].reshape(hidden_size), net.nodes[5].slot)
	if not np.allclose(a, b):
		print('diff: %f' % np.sum((a - b)**2))
		raise Exception('final hidden state mismatch')

	print('[ ok ] backward test passed')

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


n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by)  # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length  # loss at iteration 0
for i in range(2):  # while True:
	print('stage %d:' % i)
	# prepare inputs (we're sweeping from left to right in steps seq_length long)
	if p+seq_length+1 >= len(data) or n == 0:
		hprev = istate  # np.zeros((hidden_size, 1))  # reset RNN memory
		p = 0  # go from start of data
	inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
	targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

	# sample from the model now and then
	if n % 100 == 0:
		sample_ix = sample(hprev, inputs[0], 200)
		txt = ''.join(ix_to_char[ix] for ix in sample_ix)
		# print('----\n %s \n----' % (txt, ))

	# forward seq_length characters through the net and fetch gradient
	loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)

	smooth_loss = smooth_loss * 0.999 + loss * 0.001
	# if n % 100 == 0: print('iter %d, loss: %f' % (n, smooth_loss))  # print progress

	# perform parameter update with Adagrad
	for param, dparam, mem in zip(
		[Wxh, Whh, Why, bh, by],
		[dWxh, dWhh, dWhy, dbh, dby],
		[mWxh, mWhh, mWhy, mbh, mby]
	):
		mem += dparam * dparam
		param += -learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update

	net.push(0, 'learn')

	(a, b) = (Whh.T, net.nodes[6].weight)
	if not np.allclose(a, b):
		print('diff: %f' % np.sum((a - b)**2))
		raise Exception('learn Why weight mismatch')
	print('[ ok ] learn test passed')

	p += seq_length  # move data pointer
	n += 1  # iteration counter
