import matplotlib.pylab as plt
import numpy as np

def activeFunctionCalling():
    # x = np.arange(-8, 8, 0.1)
    f = 1 / (1 + np.exp(-x))
    plt.plot(x, f)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()
    print(f)

def baisFunctionCalling():
    w1 = 0.5
    w2 = 1.0
    w3 = 2.0
    l1 = 'w = 0.5'
    l2 = 'w = 1.0'
    l3 = 'w = 2.0'
    for w, l in [(w1, l1), (w2, l2), (w3, l3)]:
        f = 1 / (1 + np.exp(-x * w))
        plt.plot(x, f, label=l)
    plt.xlabel('x')
    plt.ylabel('h_w(x)')
    plt.legend(loc=2)
    plt.show()
    print(1/(1+np.exp(-1))) ##0.7310585786300049
    print(1/(1+np.exp(-2))) ##0.8807970779778823
    print(1/(1+np.exp(-5))) ##0.9933071490757153


def biasFunctionCalling2withweightage():
    w = 5.0
    b1 = -8.0
    b2 = 0.0
    b3 = 8.0
    l1 = 'b = ‐8.0'
    l2 = 'b = 0.0'
    l3 = 'b = 8.0'
    for b, l in [(b1, l1), (b2, l2), (b3, l3)]:
        f = 1 / (1 + np.exp(-(x * w+b)))
        plt.plot(x, f, label=l)
    plt.xlabel('x')
    plt.ylabel('h_wb(x)')
    plt.legend(loc=2)
    plt.show()

def f(x):
    return 1 / (1 + np.exp(-x))



def simple_looped_nn_calc(n_layers, x, w, b):
    for l in range(n_layers-1):
        #Setup the input array which the weights will be multiplied by for each layer
        #If it's the first layer, the input array will be the x input vector
        #If it's not the first layer, the input to the next layer will be the
        #output of the previous layer
        if l == 0:
            node_in = x
        else:
            node_in = h
        #Setup the output array for the nodes in layer l + 1
        h = np.zeros((w[l].shape[0],))
        #loop through the rows of the weight array
        for i in range(w[l].shape[0]):
            #setup the sum inside the activation function
            f_sum = 0
            #loop through the columns of the weight array
            for j in range(w[l].shape[1]):
                f_sum += w[l][i][j] * node_in[j]
            #add the bias
            f_sum += b[l][i]
            #finally use the activation function to calculate the
            #i‐th output i.e. h1, h2, h3
            h[i] = f(f_sum)
    return h

def practical_2():
    x = np.arange(-5,5,0.1)
    f = 1/(1 + np.exp(-x))
    plt.plot(x, f)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()

def practical_3():
    x = np.arange(-5, 5, 0.1)
    w1=0.5
    w2=1.0
    w3=2.0
    l1 = 'b = 0.5'
    l2 = 'b = 1.0'
    l3 = 'b = 2.0'

    for w , l in [(w1,l1),(w2,l2),(w3,l3)]:
        f = 1/(1+np.exp(-x*w))
        plt.plot(x,f, label=l)
    plt.xlabel('x')
    plt.ylabel('h_w(x)')
    plt.legend(loc=2)
    plt.show()
if __name__ == '__main__':
    x = np.arange(-10, 10, 0.1)
    print("Hello World")
    activeFunctionCalling()
    baisFunctionCalling()
    biasFunctionCalling2withweightage()
    w1 = np.array([[0.2, 0.2, 0.2], [0.4, 0.4, 0.4], [0.6, 0.6, 0.6]])
    print(w1)
    w2 = np.zeros((1, 3))
    w2[0, :] = np.array([0.5, 0.5, 0.5])
    print(w2)
    b1 = np.array([0.8, 0.8, 0.8])
    b2 = np.array([0.2])
    print(b1, b2)
    w = [w1, w2]
    b = [b1, b2]
    # a dummy x input vector
    x = [1.5, 2.0, 3.0]
    sample=simple_looped_nn_calc(3,x,w,b)
    print(sample)
    practical_2()
    practical_3()

