import numpy as np
import scipy.io
from nn import *
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import string

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36
learning_rate = 3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# Q5.1 & Q5.2
# initialize layers here
##########################
##### your code here #####
##########################

initialize_weights(train_x.shape[1], hidden_size, params, "input")
initialize_weights(hidden_size, hidden_size, params, "hidden1")
initialize_weights(hidden_size, hidden_size, params, "hidden2")
initialize_weights(hidden_size, train_x.shape[1], params, "output")
params['m_Winput'] = np.zeros_like(params['Winput'])
params['m_binput'] = np.zeros_like(params['binput'])
params['m_Whidden1'] = np.zeros_like(params['Whidden1'])
params['m_bhidden1'] = np.zeros_like(params['bhidden1'])
params['m_Whidden2'] = np.zeros_like(params['Whidden2'])
params['m_bhidden2'] = np.zeros_like(params['bhidden2'])
params['m_Woutput'] = np.zeros_like(params['Woutput'])
params['m_boutput'] = np.zeros_like(params['boutput'])


# should look like your previous training loops
losses = []
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions

        ##########################
        ##### your code here #####
        ##########################
        h1 = forward(xb, params, 'input', relu)
        h2 = forward(h1, params, 'hidden1', relu)
        h3 = forward(h2, params, 'hidden2', relu)
        prob = forward(h3, params, 'output', sigmoid)
        loss = np.sum((prob - xb) ** 2)
        total_loss = total_loss + loss
        delta1 = backwards(2 * (prob - xb), params, 'output', sigmoid_deriv)
        delta2 = backwards(delta1, params, 'hidden2', relu_deriv)
        delta3 = backwards(delta2, params, 'hidden1', relu_deriv)
        backwards(delta3, params, 'input', relu_deriv)

        params['m_W' + 'input'] = 0.9 * params['m_W' + 'input'] - learning_rate * params['grad_W' + 'input']
        params['W' + 'input'] += params['m_W' + 'input']
        params['m_b' + 'input'] = 0.9 * params['m_b' + 'input'] - learning_rate * params['grad_b' + 'input']
        params['b' + 'input'] += params['m_b' + 'input']

        params['m_W' + 'hidden1'] = 0.9 * params['m_W' + 'hidden1'] - learning_rate * params['grad_W' + 'hidden1']
        params['W' + 'hidden1'] += params['m_W' + 'hidden1']
        params['m_b' + 'hidden1'] = 0.9 * params['m_b' + 'hidden1'] - learning_rate * params['grad_b' + 'hidden1']
        params['b' + 'hidden1'] += params['m_b' + 'hidden1']

        params['m_W' + 'hidden2'] = 0.9 * params['m_W' + 'hidden2'] - learning_rate * params['grad_W' + 'hidden2']
        params['W' + 'hidden2'] += params['m_W' + 'hidden2']
        params['m_b' + 'hidden2'] = 0.9 * params['m_b' + 'hidden2'] - learning_rate * params['grad_b' + 'hidden2']
        params['b' + 'hidden2'] += params['m_b' + 'hidden2']

        params['m_W' + 'output'] = 0.9 * params['m_W' + 'output'] - learning_rate * params['grad_W' + 'output']
        params['W' + 'output'] += params['m_W' + 'output']
        params['m_b' + 'output'] = 0.9 * params['m_b' + 'output'] - learning_rate * params['grad_b' + 'output']
        params['b' + 'output'] += params['m_b' + 'output']

    losses.append(total_loss/train_x.shape[0])
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9

# plot loss curve
plt.plot(range(len(losses)), losses)
plt.xlabel("epoch")
plt.ylabel("average loss")
plt.xlim(0, len(losses)-1)
plt.ylim(0, None)
plt.grid()
plt.show()

        
# Q5.3.1
# choose 5 labels (change if you want)
visualize_labels = ["A", "B", "C", "1", "2"]

# get 2 validation images from each label to visualize
visualize_x = np.zeros((2*len(visualize_labels), valid_x.shape[1]))
for i, label in enumerate(visualize_labels):
    idx = 26+int(label) if label.isnumeric() else string.ascii_lowercase.index(label.lower())
    choices = np.random.choice(np.arange(100*idx, 100*(idx+1)), 2, replace=False)
    visualize_x[2*i:2*i+2] = valid_x[choices]

# run visualize_x through your network
# name the output reconstructed_x
##########################
##### your code here #####
##########################
h1 = forward(visualize_x, params, 'input', relu)
h2 = forward(h1, params, 'hidden1', relu)
h3 = forward(h2, params, 'hidden2', relu)
reconstructed_x = forward(h3, params, 'output', sigmoid)


# visualize
fig = plt.figure()
plt.axis("off")
grid = ImageGrid(fig, 111, nrows_ncols=(len(visualize_labels), 4), axes_pad=0.05)
for i, ax in enumerate(grid):
    if i % 2 == 0:
        ax.imshow(visualize_x[i//2].reshape((32, 32)).T, cmap="Greys")
    else:
        ax.imshow(reconstructed_x[i//2].reshape((32, 32)).T, cmap="Greys")
    ax.set_axis_off()
plt.show()


# Q5.3.2
from skimage.metrics import peak_signal_noise_ratio
# evaluate PSNR
##########################
##### your code here #####
##########################
psnr = 0
for i in range(visualize_x.shape[0]):
    psnr += peak_signal_noise_ratio(visualize_x[i], reconstructed_x[i])
psnr = psnr/visualize_x.shape[0]
print("PSNR: ", psnr)
