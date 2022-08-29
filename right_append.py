
import sys

import time
import random

from networkx.drawing.nx_pydot import graphviz_layout
from data_handling import data_augmentation, data_loading, data_split, Image
from layer_generation import layer_gen
from network import Network, Perceptron





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

    random.seed(1)
    t_images_path = "mnist dataset\\training-images.txt"
    t_labels_path = "mnist dataset\\training-labels.txt"

    (t_images, rows, cols, digits) = data_loading(t_images_path, t_labels_path)
    (train_set, test_set) = data_split(t_images, 0.75)
    

    

    train_sets = data_augmentation(train_set, 10)

    output_layer: "list[Perceptron]" = []

    #perceptrons = layer_gen(train_sets[0] + train_sets[1] + train_sets[2], 28*28)

    #for i in digits:
    #    output_layer.append(Perceptron(None, None, len(perceptrons)))

    #net = Network([perceptrons, output_layer])

    for i in digits:
        output_layer.append(Perceptron(None, None, rows * cols))
    net = Network([output_layer])

    

    print("Network built, starting training")

    start = time.time()

    grow_times = 3
    repeats = 10

    if len(sys.argv) > 1:
        repeats = int(sys.argv[1])
    if len(sys.argv) > 2:
        grow_times = int(sys.argv[2])

    grow_index = int(repeats / grow_times)

    #first_err = tot_err = net.train(train_set)
    #print(first_err)
    #prev_diff = 0

    for i in range(0,repeats):
        train_set_i = random.randint(0, len(train_sets) - 1)
        tot_err = net.train(train_sets[train_set_i])
        #tot_err = net.train(train_set)
        print("total error on training:", tot_err)

        if i % grow_index == grow_index - 1:
            net.test(test_set)

            if i <= repeats - 5:
                net.grow_network()


        # curr_diff = first_err / tot_err

        # if curr_diff - prev_diff  < 0.1:
        #     print(first_err)
        #     print(prev_diff)
        #     print(curr_diff)
        #     net.grow_network()
        #     first_err = net.train(train_set)
        #     curr_diff = 0

        # #print("-------", curr_diff - prev_diff, "-------")
        # #print(curr_diff, prev_diff)

        # prev_diff = curr_diff

    end = time.time()

    print("Time:", end - start, "seconds")
    net.show_net()

    return

run()
 