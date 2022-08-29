
import numpy as np
import math
import random

import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
from data_handling import data_augmentation, data_loading, data_split, Image

INPUT_LAYER = 0
OUTPUT_LAYER = 100001
VERBOSE = True

'''
A class representing individual perceptrons, potentially superflous
'''
class Perceptron():

    '''
    param network - network it's connected to
    param layers - a dictionary of the layerId and np.array of floats, can be None for 1st layer
    param in_nodes - number of inputs to layer, for 1st layer
    #! Should separate into 2 functions, first_layer() and from_existing()
    '''
    def __init__(self, network, layers: "dict[int, list[float]]", in_nodes):

        if layers != None:
            self.layers = layers
        else: #Generate the weights randomly
            w = [0.1 * random.random() - 0.05 for x in range(in_nodes)]

            for i, weight in enumerate(w): #Makes sure no weights are init to true zero
                if weight == 0:
                    w[i] += 0.01
            self.layers = {0:np.array(w)}

        self.network = network
        self.output = 0

    '''
    Used to get the weights of a specific layer, if it exists
    '''
    def get_connections_layer(self, layer_id):
        return self.layers.get(layer_id, [])

    '''
    Sets the network the node is connected to
    '''
    def set_network(self, network):
        self.network = network

    '''
    Extends a layer by increasing the size to layer_len, new weights are randomly initialized
    param layer_id - id of layer to extend
    param layer_len - new length of the layer
    '''
    def extend_layer(self, layer_id, layer_len):

        if layer_id not in self.layers: #If layer doesn't exist, randomly gen layer and return
            self.layers[layer_id] = np.array([0.1 * random.random() - 0.05 for _ in range(layer_len)])
            return

        weights = self.layers[layer_id]

        if len(weights) < layer_len: # Generate new layer and copy over existing weights
            new_weights = np.array([0.1 * random.random() - 0.05 for _ in range(layer_len)])
            for i, val in enumerate(weights):
                new_weights[i] = val
            self.layers[layer_id] = new_weights

    '''
    Adds a child to this node by extending the size or number of layers if necessary
    param weight - weight to new child
    param weight_index - index of child in layer
    param layer_id - layer id to add the child to
    param layer_size - size of the layer
    '''
    def add_child(self, weight, weight_index, layer_id, layer_size):

        self.extend_layer(layer_id, layer_size)
        layer = self.layers[layer_id]
        layer[weight_index] = weight

    '''
    Removes a layer and all the weights to it
    '''
    def remove_layer(self, layer_id):
        del self.layers[layer_id]

    '''
    #!Not used
    Basically copies itself into a child node
    param layer_id - layer id of new child
    param layer_size - layer size of new child
    '''
    def spawn_child(self, layer_id, layer_size):
        #TODO: Should I remove the layer completely from the parent, saving a lot of computation.

        child_layers = {} #Create the childs layers dictionary

        # Copies each layer exactly but skips certain "weak" connections 
        for l_id, layer in self.layers.items():

            child_weights = layer.copy()
        
            for i, w in enumerate(child_weights):

                if w < 0.1 and w > -0.1:
                    child_weights[i] = 0
                else: # Disconnects itself from the input layer
                    if l_id == 0:
                        layer[i] = 0

            child_layers[l_id] = child_weights

        
        child = Perceptron(self.network, child_layers, -1) # Creates the child


        #Creates the child layer and connects this Perceptron to the child
        new_layer = np.array([0.1 * random.random() - 0.05 for _ in range(layer_size)])
        new_layer[-1] = 0.9
        self.layers[layer_id] = new_layer
        return child


    '''
    #!Not used
    Calculates if this Perceptron should spawn a child.
    Checks if there is a certain proportion of strong connections to active connections
    '''
    def should_spawn(self):

        strong = 0.3

        strong_req = 0.1

        minimal_req = 30
        active_connections = 0
        strong_connections = 0

        for _, layer in self.layers.items():

            active_connections += len([w for w in layer if w != 0])
            strong_connections += len([w for w in layer if abs(w) > strong])
            #print([w for w in layer if abs(w) > strong])
        
        if strong_connections > active_connections * strong_req and active_connections > minimal_req:
            return True
        else:
            return False

    '''
    Calculates the local backpropagation for this Perceptron.
    param in_error - error of this Perceptrons estimation
    param next_layer - next layer to calulate errors for
    returns - If the next_layer is not input layer then it returns the error from this Perceptron on the next layer
    '''
    def backprop_ich(self, in_error, next_layer):
        lr = 0.1
        
        error = in_error * (Perceptron.transfer_derivative(self.output))
        next_errors = []

        for key, layer in self.layers.items():

            for i, pixel in enumerate(self.network.get_layer(key)):

                if layer[i] != 0:
                    layer[i] += lr * error * pixel

        if next_layer != INPUT_LAYER and next_layer in self.layers:
            next_errors = self.layers[next_layer] * error


        return (error, next_errors)

    '''
    Calculates the Output of this Perceptron
    '''
    def activate(self):
        activation = 0

        #Extends any layer if needed
        for layer_id, weights in self.layers.items():
            layer_outputs = self.network.get_layer(layer_id)

            if len(weights) < len(layer_outputs):
                self.extend_layer(layer_id, len(layer_outputs))
                weights = self.layers[layer_id]
            
            activation += np.dot(weights, layer_outputs)

        self.output = self.activation(activation)
        return self.output

    '''
    Activation function, sigmoid.
    param: x - input.
    '''
    def activation(self, x):
        return 1 / (1 + math.exp(-x))

    
    
    # Calculate the derivative of an neuron output
    def transfer_derivative(output):
	    return output * (1.0 - output)



class Network():

    def __init__(self, layers: "list[list[Perceptron]]"):

        self.layers = {OUTPUT_LAYER:layers[-1]}
        self.layer_order = [OUTPUT_LAYER]
        self.layer_outputs = {OUTPUT_LAYER: np.zeros(len(layers[-1]))}

        for i, layer in enumerate(layers[0:-1]):

            new_layer_id = OUTPUT_LAYER

            while new_layer_id in self.layers: # Make sure new layer_id is unique
                new_layer_id = random.randint(1, 100000)

            self.layers[new_layer_id] = layer
            self.layer_order = [new_layer_id] + self.layer_order
            self.layer_outputs[new_layer_id] = np.zeros(len(layer))

            if i < len(layers):
                for perceptron in layers[i + 1]:    
                    perceptron.extend_layer(new_layer_id, len(layer))

        
        for layer in layers:
            for perceptron in layer:
                perceptron.set_network(self)
        

        if len(layers) > 1:
            for perceptron in layers[-1]:
                    perceptron.remove_layer(0)

        self.image = None
        self.classifier_map = {0:4, 1:7, 2:8, 3:9}
        #self.layers.extendleft([["Test1.1", "Test1.2"]])

    '''
    Copies the non-common strong connections of two nodes into a child node
    '''
    def add_uncommon_percp(self, child_l_id, parent1: Perceptron, parent2:Perceptron):
        child_layers = {}

        for layer_id in [0] + self.layer_order:

            if layer_id == child_l_id:
                break

            conn1 = parent1.get_connections_layer(layer_id)
            conn2 = parent2.get_connections_layer(layer_id)

            if len(conn1) != 0 and len(conn2) != 0:
                child_weights = conn1.copy()

                for i, w in enumerate(child_weights):

                    w = (w - conn2[i]) / 2

                    # if abs(w) < 0.025:
                    #     child_weights[i] = 0
                    # else:
                    #     child_weights[i] = w

                child_layers[layer_id] = child_weights
                #print("added", len(child_weights), "to", l_id)

        child = Perceptron(self, child_layers, -1)
        self.layers[child_l_id].append(child)


        #return (parent1_w, parent2_w)


    '''
    Makes a child node for each potential grouping of the parent.
    Either child node is made to prioritize common or uncommon connections
    '''
    def all_common_layer(self, child_l_id, parent_l_id):
        new_layer = []

        if VERBOSE == True:
                print("growing with layer: ", child_l_id)
        # Setup
        self.layers[child_l_id] = new_layer
        self.layer_outputs[child_l_id] = np.zeros(len(new_layer))
        new_lo = []

        for lo in self.layer_order:
            
            if lo == parent_l_id:
                new_lo.append(child_l_id)
            new_lo.append(lo)
        self.layer_order = new_lo

        # Loop parent combinations 1-2, 1-3, 1-4, 2-3, 2-4, 3-4
        parent_layer = self.layers[parent_l_id]
        prev_len_child_l = len(self.layers[child_l_id])

        for i, parent1 in enumerate(parent_layer):

            if i == len(parent_layer) - 1:
                break

            for parent2 in parent_layer[i+1:]:

                #self.add_common_perceptron(child_l_id, parent1, parent2)
                self.add_uncommon_percp(child_l_id, parent1, parent2)

        new_len_child_l = len(self.layers[child_l_id])
        for i, parent1 in enumerate(parent_layer): #This is done later to not extend the parents array several times

            if i == len(parent_layer) - 1:
                break
            
            for j, parent2 in enumerate(parent_layer[i+1:]):
                
                #Adjust weights to new childs if wanted.
                parent2.add_child(0.5, prev_len_child_l + i + j, child_l_id, new_len_child_l)
                parent1.add_child(0.5, prev_len_child_l + i + j, child_l_id, new_len_child_l)

        self.layer_outputs[child_l_id] = np.zeros(new_len_child_l)
        


    '''
    Copies the common strong connections of two nodes into a child node
    '''
    def add_common_perceptron(self, child_l_id, parent1:Perceptron, parent2:Perceptron):
        child_layers = {}

        for layer_id in [0] + self.layer_order:

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

                    # if abs(w) < 0.025:
                    #     child_weights[i] = 0
                    # else:
                    #     child_weights[i] = w

                child_layers[layer_id] = child_weights
                #print("added", len(child_weights), "to", l_id)

        child = Perceptron(self, child_layers, -1)
        self.layers[child_l_id].append(child)

    '''
    Copies the strong connections of a node into a child node
    '''
    def add_childs(self, new_layer_id, layer_id):

        layer = self.layers[layer_id]

        new_layer = []
            
        for perceptron in layer:

            if perceptron.should_spawn():
                
                new_layer.append(perceptron.spawn_child(new_layer_id, len(new_layer) + 1))
                #perceptrons_that_spawned.append(perceptron)

        if len(new_layer) > 0 :

            if VERBOSE == True:
                print("growing with layer: ", new_layer_id)

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

    '''
    Grow the network
    Can either copy existing nodes
    Or make merge between two nodes alike weights
    Or make merge between two nodes unalike weights
    Generates only from the output layer at the moment, removes connections to earlier layers as well
    '''
    def grow_network(self):

        if VERBOSE == True:
            print("potential growing")

        new_layer_id = OUTPUT_LAYER

        while new_layer_id in self.layers: # Make sure new layer_id is unique
            new_layer_id = random.randint(1, 100000)

        #? What type of layer to add
        #self.add_childs(new_layer_id, layer_id)
        self.all_common_layer(new_layer_id, OUTPUT_LAYER)

        # removes connections to earlier layers
        rem_layer = 0
        if len(self.layer_order) > 2:
            rem_layer = self.layer_order[-3]
        
        for out_percp in self.layers[OUTPUT_LAYER]:
            out_percp.remove_layer(rem_layer)

        #print("added a total of ", len(self.layers[new_layer_id]) , "new nodes")
        #print("After", self.layer_order)
            
    '''
    Returns a specific layer, unless layer_id == 0 then return the Image instead
    '''
    def get_layer(self, layer_id):

        if layer_id == INPUT_LAYER:
            return self.image
        return self.layer_outputs[layer_id]

    '''
    Train the network on a set of Images
    returns - total error for all images and percentage of correctly classified images
    '''
    def train(self, images: "list[Image]"):

        tot_err = 0
        correct = 0

        for image in images:
            highest_activation = 0
            highest_activation_i = 0
            image_err = 0
            self.image = image.get_pixels()

            # Predict layer for layer
            for lo in self.layer_order:
                outputs = self.layer_outputs[lo]
                
                for i, perceptron in enumerate(self.layers[lo]):
                    outputs[i] = perceptron.activate()

            # Calculate if the output is correct
            for i, output in enumerate(self.layer_outputs[OUTPUT_LAYER]):

                if output > highest_activation:
                    highest_activation_i = i
                    highest_activation = output

            if self.classifier_map[highest_activation_i] == image.get_label():
                correct += 1

            #print(self.classifier_map[highest_activation_i], image.get_label(), self.layer_outputs[OUTPUT_LAYER])

            image_err = self.backprop(image)
            tot_err += abs(image_err)
        return (tot_err, correct / len(images))

    '''
    Backpropagation for a specific Image
    '''
    def backprop(self, image: Image):

        next_layer = 0
        next_errors = None

        # If there are more than 1 layer we use backprop, otherwise GC
        if len(self.layer_order) > 1:
            next_layer = self.layer_order[-2]
            next_errors = np.zeros(len(self.layers[next_layer]))

        image_err = 0

        output_layer = self.layer_outputs[OUTPUT_LAYER]

        # GC output layer and calculate error between output and expected output
        for i, out_perceptrons in enumerate(self.layers[OUTPUT_LAYER]):
            expected = 1 if image.get_label() == self.classifier_map[i] else 0
            percp_error = expected - output_layer[i]

            (error, next_errors_percp) = out_perceptrons.backprop_ich(percp_error, next_layer)
            
            image_err += error

            # If not 1st layer, add the error from this Perceptron to next layer
            if len(next_errors_percp) != 0:
                next_errors += next_errors_percp

        # GC for the ith layer (reversed order, from output-1 to first layer) and calculate error for next layer
        for i, lo in enumerate(reversed(self.layer_order[0:-1])):
            i = len(self.layer_order) - i - 2 #Reverse the i 
            curr_errors = next_errors

            if i > 0:
                next_layer = self.layer_order[i - 1]
                next_errors = np.zeros(len(self.layers[next_layer])) #!Inefficient, should add to net as a static array to avoid uncessary malloc
            else:
                next_layer = 0            
            
            # GC and error calc
            for i, out_perceptrons in enumerate(self.layers[lo]):
                error = curr_errors[i] if len(curr_errors) > i else 0
                (_, next_errors_percp) = out_perceptrons.backprop_ich(curr_errors[i], next_layer)

                # If not 1st layer, add the error from this Perceptron to next layer
                if len(next_errors_percp) != 0:
                    next_errors += next_errors_percp

            
        return image_err


    '''
    Tests the network on a set of Images
    returns - number of correct guesses, highest error on guess, avg highest error 
    '''
    def test(self, images: "list[Image]"):
        correct = 0
        highest_error = 0
        avg_highest_error = 0

        for image in images:

            #Select image
            self.image = image.get_pixels()

            #Run the image through the net going from lower to higher layers
            for lo in self.layer_order:
                outputs = self.layer_outputs[lo]
                
                for i, perceptron in enumerate(self.layers[lo]):
                    outputs[i] = perceptron.activate()

                self.layer_outputs[lo] = outputs

            #Check if it classifies the image correctly
            highest_error_img = 0
            highest_activation = 0 
            highest_activation_i = 0
            for i, output in enumerate(self.layer_outputs[OUTPUT_LAYER]):
                expected = 1 if image.get_label() == self.classifier_map[i] else 0
                #Find highest error
                highest_error_img = max(abs(output - expected), highest_error_img)

                if output > highest_activation:
                    highest_activation_i = i
                    highest_activation = output

            if self.classifier_map[highest_activation_i] == image.get_label():
                correct += 1

            highest_error = max(highest_error_img, highest_error)
            avg_highest_error += highest_error_img

        print("Average highest error:", avg_highest_error / len(images))
        print("Highest error:", highest_error)
        print("Correct percentage: ", correct / len(images))
        print("Total images:", len(images))          

    '''
    Shows the network as a tree
    Uses NetworkX and matplotlib
    param incl_input - Should the input layer be included
    #? Pretty clunky solution, probably easier to render yourself, especially if you want real-time and weights with colours
    '''
    def show_net(self, incl_input=False):
        total_nodes = 0
        in_nodes = 28 * 28

        for layer_id, layer in self.layers.items():
            total_nodes += len(layer)
        
        neigh_matrix = np.zeros((total_nodes, total_nodes))
        i1 = 0

        if incl_input:
            neigh_matrix = np.zeros((total_nodes + in_nodes, total_nodes + in_nodes))
            i1 = in_nodes

        for i, layer_id in enumerate(self.layer_order):

            layer = self.layers[layer_id]

            for perceptron in layer:

                i2 = 0
                layers = self.layer_order[:i]

                if incl_input:
                    i2 = in_nodes
                    layers = [0] + self.layer_order[:i]
                
                for l_id2 in layers:
                    connections = perceptron.get_connections_layer(l_id2)

                    if len(connections) == 0:
                        i2 += len(self.layers[l_id2])

                    for conn in connections:

                        if abs(conn) > 0.05:
                        #if conn != 0:
                            neigh_matrix[i1][i2] = 1
                            neigh_matrix[i2][i1] = 1
                        i2 += 1
                i1 += 1


        # Using NetworkX to generate tree network
        G = nx.from_numpy_matrix(neigh_matrix)
        G.edges(data=True)
        labeldict = {}
        index = 0

        for i in self.layer_order:

            for j, _ in enumerate(self.layers[i]):
                labeldict[index] = str(i) + ":" + str(j)
                index += 1
        
        pos = graphviz_layout(G, prog="dot")
        nx.draw(G, pos=pos, labels=labeldict, with_labels = True)

        plt.show()
