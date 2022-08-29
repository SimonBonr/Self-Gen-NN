
#For every pixel in every image (subset of images)
#Find pixels that fire either similarly, or only pixels that fire when this fires, decide group size based on hyperparam, e.g. 10, meaning next layer is 784/10 = 78 nodes
#Then for each pixel to each pixel calculate the relative importance
    #The relative importance is calculated by looking at 2 pixels and trying to prioritize their unique points

from data_handling import Image
import numpy as np
import matplotlib.pyplot as plt

def layer_gen(images: "list[Image]", n_pixels: int):

    common_outputs: "dict[int][dict[int][(float, int)]]" = {}
    
    for image in images:

        for i, pixel_val in enumerate(image.get_pixels()):

            if pixel_val > 0.01:

                if i not in common_outputs:
                    common_outputs[i] = {}
                    #print(i, "gets neighbours")

                for j, pixel2_val in enumerate(image.get_pixels()):

                    if i == j and pixel2_val < 0.01:
                        continue
                    
                    if j not in common_outputs[i]:
                        common_outputs[i][j] = (pixel2_val, 1)
                    else:
                        prev, occur = common_outputs[i][j]
                        common_outputs[i][j] = (prev + pixel2_val, occur + 1)


    tot_hits_pixels = {}

    for i in range(0,n_pixels):

        if i not in common_outputs:
            continue
        tot_hits = 0

        for j in range(0,n_pixels):
            
            if j in common_outputs[i]:
                total, occur = common_outputs[i][j]
                common_outputs[i][j] = (total / occur, occur)
                tot_hits += occur
        tot_hits_pixels[i] = tot_hits


    print(tot_hits_pixels[406])
    
    show_as_image(common_outputs, n_pixels, 406) # 406, 714
    
    overlaps: "dict[int][dict[int][int]]" = {}

    #!PRETTY SURE THIS SHIT STILL DOESN'T WORK
    #for i in range(0, n_pixels):
    for i in range(406, 407):

        if i not in common_outputs:
            continue

        overlaps[i] = {}

        for j in range(0,n_pixels):

            if i == j:
                continue
            if j not in common_outputs:
                #print("ASKJDHKAJSHDKJASHDKJASHDKJAHSDJKASDHKJASHD")
                continue

            #for p_i2, _ in common_outputs[i].items():
            for p_i2 in range(0,n_pixels):

                if p_i2 not in common_outputs[i]:
                    continue

                if p_i2 in common_outputs[j]:

                    if p_i2 not in overlaps[i]:
                        overlaps[i][p_i2] = common_outputs[j][p_i2][1] * common_outputs[j][p_i2][0]
                    else:
                        overlaps[i][p_i2] += common_outputs[j][p_i2][1] * common_outputs[j][p_i2][0]

    show_as_image(overlaps, n_pixels, 406) # 406, 714

    #for i in range(0, n_pixels):
    for i in range(406, 407):

        if i not in common_outputs:
            continue

        for j in range(0,n_pixels):
            
            if j in common_outputs[i]:
                avg, occur = common_outputs[i][j]
                #avg = avg - avg * (overlaps[i][j] / tot_hits_pixels[i])
                avg = avg - avg * (overlaps[i][j] / (tot_hits_pixels[i] / 4)) #!Make this value the max value thing
                common_outputs[i][j] = (avg, occur)

    show_as_image(common_outputs, n_pixels, 406) # 406, 714
        
            


def show_as_image(common_outputs, size, pixel_i):

    img = np.zeros(size)

    for j in range(0, size):

        if j not in common_outputs[pixel_i]:
            continue

        if j == pixel_i:
            img[j] = 0.255
        else:
            try:
                img[j] = common_outputs[pixel_i][j][0]
            except:
                img[j] = common_outputs[pixel_i][j]
                print(j, img[j])

    fig = plt.figure(figsize=(10, 1))
    ax1 = fig.subplots(1, 1)

    ax1.imshow(img.reshape(28,28), cmap='gray')
    fig.set_tight_layout(True)
    plt.show()


