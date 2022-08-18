from imp import new_module
import sys
import numpy as np
import time
import math
from threading import Thread
import random
#import llist
#from llist import sllist,sllistnode
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout



'''
Image class based of pixels as ascii values from a text file.
Also stores the label for the image if available.
'''
class Image():

    '''
    Initializes class
    param: imagefile - File to get ascii pixels from. One image per line
    param: labelfile, default: none - File with label for image. One label per line.
    '''
    def __init__(self, imagefile, labelfile=None):
        pixels = list(map(int,imagefile.readline().split()))
        pixels = list(map((0.001).__mul__,pixels))
        self.pixels = np.array(pixels)
        #super().set_nodes(pixels)

        if labelfile != None:
            self.label = int(labelfile.readline())

    def get_pixels(self):
        return self.pixels
    
    '''
    Returns label for image if available
    '''
    def get_label(self):
        return self.label


class Perceptron():

    def __init__(self, network, layers: "dict[int, list[float]]", in_nodes):

        if layers != None:
            self.layers = layers
        else:
            w = [0.1 * random.random() - 0.05 for x in range(in_nodes)]

            for i, weight in enumerate(w): #Makes sure no weights are init to true zero
                if weight == 0:
                    w[i] += 0.01
            self.layers = {0:np.array(w)}

        #self.layer = layer
            
        #self.in_nodes = in_nodes
        self.network = network
        self.output = 0

    def get_connections_layer(self, layer_id):
        return self.layers.get(layer_id, [])

    def set_network(self, network):
        self.network = network

    def extend_layer(self, layer_id, layer_len):

        if layer_id not in self.layers:
            self.layers[layer_id] = np.array([0.1 * random.random() - 0.05 for x in range(layer_len)])
            return

        weights = self.layers[layer_id]

        if len(weights) < layer_len:
            new_weights = np.array([0.1 * random.random() - 0.05 for x in range(layer_len)])
            for i, val in enumerate(weights):
                new_weights[i] = val
            self.layers[layer_id] = new_weights

    def add_child(self, weight, weight_index, layer_id, layer_size):

        self.extend_layer(layer_id, layer_size)
        layer = self.layers[layer_id]
        layer[weight_index] = weight


    def spawn_child(self, layer_id, layer_size):
        #TODO: Should I remove the layer completely from the parent, saving a lot of computation.

        child_layers = {}

        for l_id, layer in self.layers.items():

            child_weights = layer.copy()
        
            for i, w in enumerate(child_weights):

                if w < 0.1 and w > -0.1:
                    child_weights[i] = 0
                else:
                    if l_id == 0:
                        layer[i] = 0

            child_layers[l_id] = child_weights
            #print("added", len(child_weights), "to", l_id)

        child = Perceptron(self.network, child_layers, -1)


        new_layer = np.array([0.1 * random.random() - 0.05 for x in range(layer_size)])
        new_layer[-1] = 0.9
        self.layers[layer_id] = new_layer
        return child


    def should_spawn(self):

        strong = 0.5

        strong_req = 0.1

        minimal_req = 30
        active_connections = 0
        strong_connections = 0

        for _, layer in self.layers.items():

            active_connections += len([w for w in layer if w != 0])
            strong_connections += len([w for w in layer if abs(w) > strong])
        
        if strong_connections > active_connections * strong_req and active_connections > minimal_req:
            return True
        else:
            return False

    def backprop_ich(self, label, classifying):
        lr = 0.1
        #TODO: Start by just adjusting layer 0 weights and try without ever growing the net to make sure that it works before anything
        expect = 1 if label == classifying else 0
        #print("class: " ,self.classifier, ", label: ", image.get_label(), ", expect: ", expect, ", res: ", res)
        error = expect - self.output

        #in_layer = self.layers[0]

        for key, layer in self.layers.items():

            for i, pixel in enumerate(self.network.get_layer(key)):

                if layer[i] != 0:
                    layer[i] += lr * error * pixel

        return error

    def activate(self):
        activation = 0

        for layer_id, weights in self.layers.items():
            layer_outputs = self.network.get_layer(layer_id)

            if len(weights) < len(layer_outputs):
                #print(self)
                #new_weights = np.array([0.1 * random.random() - 0.05 for x in range(len(layer_outputs))])
                #print(len(layer_outputs), len(weights))
                #weights += np.concatenate([weights, new_weights], axis=None)
                #for i, val in enumerate(weights):
                #    new_weights[i] = val
                #new_weights[:len(weights)] = weights
                #self.layers[layer_id] = new_weights
                #self.layers[layer_id] = new_weights
                self.extend_layer(layer_id, len(layer_outputs))
                weights = self.layers[layer_id]

                #activation += np.dot(new_weights, layer_outputs)
            
            activation += np.dot(weights, layer_outputs)


        self.output = self.activation(activation)
        return self.output

    '''
    Activation function, sigmoid.
    param: x - input.
    '''
    def activation(self, x):
        return 1 / (1 + math.exp(-x))



class Network():

    def __init__(self, output_layer: "list[Perceptron]"):

        for perceptron in output_layer:
            perceptron.set_network(self)
        
        self.layers = {100001:output_layer}
        self.layer_order = [100001]
        self.layer_outputs = {100001: np.zeros(len(output_layer))}
        self.image = None
        self.classifier_map = {0:4, 1:7, 2:8, 3:9}
        #self.layers.extendleft([["Test1.1", "Test1.2"]])

    def add_perceptron(self, layer):

        if layer >= self.layers.size:
            print("Can't add perceptron to non-existing layer")
            assert(False)
        self.layers.nodeat(layer).append()

    def all_common_layer(self, child_l_id, parent_l_id):
        parent_layer = self.layers[parent_l_id]
        prev_len_child_l = len(self.layers[child_l_id])

        for i, parent1 in enumerate(parent_layer):
            parent2 = parent_layer[(i + 1) % len(parent_layer)]

            self.add_common_perceptron(child_l_id, parent1, parent2)

        new_len_child_l = len(self.layers[child_l_id])
        for i, parent1 in enumerate(parent_layer): #This is done later to not extend the parents array several times
            parent2 = parent_layer[(i + 1) % len(parent_layer)]

            parent1.add_child(0.5, prev_len_child_l + i, child_l_id, new_len_child_l)
            parent2.add_child(0.5, prev_len_child_l + i, child_l_id, new_len_child_l)

        self.layer_outputs[child_l_id] = np.zeros(new_len_child_l)
        


    def add_common_perceptron(self, child_l_id, parent1:Perceptron, parent2:Perceptron):
        child_layers = {}

        for layer_id in self.layer_order:

            if layer_id == child_l_id:
                break

            conn1 = parent1.get_connections_layer(layer_id)
            conn2 = parent2.get_connections_layer(layer_id)

            if len(conn1) != 0:
                child_weights = conn1.copy()
            elif len(conn2) != 0:
                child_weights = conn2.copy()
            else:
                continue
        
            if len(conn1) != 0 and len(conn2) != 0:

                for i, w in enumerate(child_weights):

                    w = (conn2[i] + w) / 2

                    if w < 0.05 and w > -0.05:
                        child_weights[i] = 0
                    else:
                        child_weights[i] = w

                child_layers[layer_id] = child_weights
                #print("added", len(child_weights), "to", l_id)

        child = Perceptron(self, child_layers, -1)
        self.layers[child_l_id].append(child)

    def add_childs(self, new_layer_id, layer_id):

        layer = self.layers[layer_id]

        new_layer = []
            
        for perceptron in layer:

            if perceptron.should_spawn():
                
                new_layer.append(perceptron.spawn_child(new_layer_id, len(new_layer) + 1))
                #perceptrons_that_spawned.append(perceptron)

        if len(new_layer) > 0 :
            #print(new_layer_id)
            #print("use------------")
            self.layers[new_layer_id] = new_layer
            self.layer_outputs[new_layer_id] = np.zeros(len(new_layer))
            new_lo = []

            for lo in self.layer_order:
                
                if lo == layer_id:
                    new_lo.append(new_layer_id)
                new_lo.append(lo)
            self.layer_order = new_lo
            return True
        return False

    def grow_network(self):

        #if len(self.layer_order) > 4:
            #return

        layer_order = self.layer_order
        #print("Before", self.layer_order)

        for layer_id in layer_order:
            #layer = self.layers[layer_id]

            new_layer_id = 100001

            while new_layer_id in self.layers:
                new_layer_id = random.randint(1, 100000)

            added = self.add_childs(new_layer_id, layer_id)

            if added:
                self.all_common_layer(new_layer_id, layer_id)
        #print("After", self.layer_order)
            

    def get_layer(self, layer):

        if layer == 0:
            return self.image
        return self.layer_outputs[layer]

    def train(self, images):

        tot_err = 0

        for image in images:
            image_err = 0
            self.image = image.get_pixels()

            for lo in self.layer_order:
                #print(lo,  flush=True)
                outputs = self.layer_outputs[lo]
                
                for i, perceptron in enumerate(self.layers[lo]):
                    outputs[i] = perceptron.activate()

                self.layer_outputs[lo] = outputs

            for i, out_perceptrons in enumerate(self.layers[self.layer_order[-1]]):
                image_err += abs(out_perceptrons.backprop_ich(image.get_label(), self.classifier_map[i]))
            #print(image_err / len(self.classifier_map))
            tot_err += image_err
        return tot_err
            
    def show_net(self):
        total_nodes = 0
        in_nodes = 28 * 28

        #print(self.layer_order)

        for layer_id, layer in self.layers.items():
            total_nodes += len(layer)
            #print(layer_id, len(layer))
        
        
        #neigh_matrix = np.zeros((total_nodes + in_nodes, total_nodes + in_nodes))
        neigh_matrix = np.zeros((total_nodes, total_nodes))

        #i1 = in_nodes
        i1 = 0
        #print(self.layer_order)

        for i, layer_id in enumerate(self.layer_order):

            layer = self.layers[layer_id]
            #print(len(self.layer_order), len(layer))

            for perceptron in layer:

                #i2 = in_nodes
                i2 = 0
                #layers = [0] + self.layer_order[:i]
                layers = self.layer_order[:i]
                #print(layers)
                for l_id2 in layers:
                    connections = perceptron.get_connections_layer(l_id2)

                    if len(connections) == 0:
                        i2 += len(self.layers[l_id2])

                    for conn in connections:

                        if abs(conn) > 0.1:
                        #if conn != 0:
                            neigh_matrix[i1][i2] = 1
                            neigh_matrix[i2][i1] = 1
                        i2 += 1
                i1 += 1

        #print(neigh_matrix)
        G = nx.from_numpy_matrix(neigh_matrix)
        #G = nx.maximum_spanning_tree(G)
        G.edges(data=True)
        labeldict = {}
        index = 0

        for i in self.layer_order:

            for j, _ in enumerate(self.layers[i]):
                labeldict[index] = str(i) + ":" + str(j)
                index += 1
        
        #pos = nx.planar_layout(G)
        #pos = graphviz_layout(G, prog="fdp")
        #pos = graphviz_layout(G, prog="acyclic")
        pos = graphviz_layout(G, prog="dot")
        nx.draw(G, pos=pos, labels=labeldict, with_labels = True)


        #nx.draw(G, labels=labeldict, with_labels = True)
        #print(G.edges())
        #print(G.nodes())
        #nx.draw(G, edges=G.edges(), width=10)
        plt.show()


    #def print_net(self):
    #    return


'''
Function run on thread that trains a perceptron on a set of images.
Then test that perceptron on a new image set and stores the total error.
'''
def run_n_test(perceptron, train_set, test_set, results, index):
    
    #perceptron.train(train_set)    
    t0 = time.time()
    for image in train_set:
        perceptron.set_nodes(image)
        res = perceptron.get_nodes()
        classif = perceptron.get_classifier()
        expect = 1 if image.get_label() == classif else 0
        error = expect - res
        errArr = [error]
        perceptron.adjust_weights(errArr, image)
    t1 = time.time()
    total = t1-t0
    print(total)

    err = perceptron.test(test_set)
    results[index] = err

'''
Main loop that runs the program.
First initializes all files and values stored in them.
Then initializes the Image objects based on the files, shuffles them and split
them into test and train set.
Then initializes the perceptrons with their classifiers.
Finally trains and tests the perceptrons to a good enough level before evaluating them.
Adjust learn rate to quickly approach 0.2 for each test and train cycle.
'''
def run():


    #init start
    #if len(sys.argv) != 4:
    #    print("Usage: python digits.py <trainings images file> <training labels file> <validation images file>")
    #    exit()

    training_images = open("mnist dataset\\training-images.txt")
    training_labels = open("mnist dataset\\training-labels.txt")
    validation_images = open("mnist dataset\\validation-images.txt")


    #traning_images = open(sys.argv[1])    
    #traning_labels = open(sys.argv[2])
    #validation_images = open(sys.argv[3])

    for i in range(0, 2):
        training_images.readline()
        training_labels.readline()
        validation_images.readline()

    [t_numi, t_rowi, t_coli, t_digi] = training_images.readline().split()
    [t_numl, t_digl] = training_labels.readline().split()
    [v_numi, v_rowi, v_coli, v_digi] = validation_images.readline().split()
    [t_numi, t_rowi, t_coli] = [int(t_numi), int(t_rowi), int(t_coli)]
    v_numi = int(v_numi)
    digits = [int(x) for x in t_digi]

    images = []
    for _ in range(0, t_numi):
        images.append(Image(training_images, training_labels))
    random.shuffle(images)

    split_i = int(t_numi * 0.75)
    train_set, test_set = images[0:split_i], images[split_i:t_numi]

    val_images = []
    for _ in range(0, v_numi):
        val_images.append(Image(validation_images))


    output_layer = []

    for i in digits:
        output_layer.append(Perceptron(None, None, t_rowi * t_coli))

    net = Network(output_layer)

    start = time.time()

    for i in range(0,30):
        tot_err = net.train(train_set)

        if i % 10 == 9:
            net.grow_network()
        print(tot_err)

    end = time.time()

    print("Time:", end - start, "seconds")
    net.show_net()

    return

    #perceptrons = []
    #for i in digits:
    #    perceptrons.append(Network(t_rowi * t_coli, i, 1, 196))
    #init end

    #Train and test perceptrons until the network has good enough accuracy.
    test_set_len = len(test_set)
    erravg = test_set_len
    loopc = 1
    threads = [None] * len(digits)
    results = [None] * len(digits)
    while erravg > 0.0885:
        errtot = 0
        #learn rate adjustion function.
        new_lr = 6/pow(2,loopc) + 0.2

        #Trains and test perceptrons on separate threads.
        for i in range(len(digits)):
            
            perceptrons[i].set_lr(new_lr)
            threads[i] = Thread(target=run_n_test, args=(perceptrons[i], train_set, test_set, results, i))
            threads[i].start()
        
        for i in range(len(digits)):
            threads[i].join()
            
        errtot = sum(results[i] for i in range(len(digits)))
        erravg = errtot / (test_set_len * 4)
        print(erravg)
        loopc += 1

    val_res = [None] * 4

    #Evaluates images by choosing the perceptron with most confident guess.
    for img in val_images:
        for i in range(len(digits)):
            val_res[i] = abs(perceptrons[i].validate(img))
        #print(digits[val_res.index(max(val_res))])
run()
 
#python3 digits.py training-images.txt training-labels.txt validation-images.txt > result.txt