import numpy as np
import random
from tokenize import String
from typing import TextIO
from scipy import ndimage, misc

'''
Image class based of pixels as ascii values from a text file.
Also stores the label for the image if available.
'''
class Image():

    
    def __init__(self, pixels, label):
        self.pixels = pixels
        self.label = label

    '''
    Initializes class
    @param: imagefile - File to get ascii pixels from. One image per line, pixel values 0-255
    @param: labelfile, default: none - File with label for image. One label per line.
    '''
    @classmethod
    def from_file(cls, imagefile: TextIO, labelfile=None):
        pixels = list(map(int,imagefile.readline().split()))
        pixels = list(map((0.001).__mul__,pixels))
        pixels = np.array(pixels)
        #super().set_nodes(pixels)
        label = ""

        if labelfile != None:
            label = int(labelfile.readline())

        return cls(pixels, label)

    '''
    Initializes Images from np.array of size (28,28) to a 784 np.array
    '''
    @classmethod
    def from_image(cls, image: np.array, label:String):
        return cls(image.reshape((28*28)), label)

    '''
    Returns the Image pixels as a np.array 
    '''
    def get_pixels(self):
        return self.pixels

    '''
    Returns the Image pixels as a np.array of shape (28,28) 
    ''' 
    def get_as_image(self):
        return self.pixels.reshape((28,28))

    '''
    Returns label for image if available
    '''
    def get_label(self):
        return self.label


'''
Uses ndimage to zoom into the image but also clips into previous size.
@param img - Image to zoom 
@param zoom_factor - factor to zoom in by
@returns - zoomed Image of same size as input
#? Taken from https://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions
'''
def __clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = ndimage.zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = ndimage.zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out




'''
Data augmentation, takes a set of images and returns n_sets of augmented images
Augmentation consists of rotation, zoom and shifting.
@param image_set - set of images to augment
@param n_sets - number of sets to augment into
@returns - n_sets of augmented image_set
'''
def data_augmentation(image_set: "list[Image]", n_sets):

    # fig = plt.figure(figsize=(10, 4))
    # ax1, ax2, ax3, ax4 = fig.subplots(1, 4)
    # img = image_set[0].get_as_image()
    # img_45 = ndimage.rotate(img, 45, reshape=False)
    # img_zoom = clipped_zoom(img, 0.9)
    # img_shift = ndimage.shift(img, -4)
    # ax1.imshow(img, cmap='gray')
    # ax1.set_axis_off()
    # ax2.imshow(img_45, cmap='gray')
    # ax2.set_axis_off()
    # ax3.imshow(img_zoom, cmap='gray')
    # ax3.set_axis_off()
    # ax4.imshow(img_shift, cmap='gray')
    # ax4.set_axis_off()
    # fig.set_tight_layout(True)
    # plt.show()

    #fig = plt.figure(figsize=(10, 2))
    #ax1, ax2 = fig.subplots(1, 2)

    train_sets = []
    #plt.ion()

    for i in range(0,n_sets):
        new_image_set: "list[Image]" = []
        fails = 0

        for image in image_set:

            rotate_val = random.random() * 40 - 20
            zoom_val = random.random() * 0.4 + 0.8
            shift_val = random.random() * 4 - 2
            #print(image.get_as_image().size)
            img_t = ndimage.rotate(image.get_as_image(), rotate_val, reshape=False)
            #print(img_t.size)
            img_t = __clipped_zoom(img_t, zoom_val)
            #print(img_t.size)
            img_t = ndimage.shift(img_t, shift_val)
            #print(img_t.size)

            if img_t.size < 784:
                new_image_set.append(image)
                fails += 1
                #exit()
                continue
            new_image_set.append(Image.from_image(img_t, image.get_label()))
            # ax1.imshow(image.get_as_image(), cmap='gray')
            # ax2.imshow(new_image_set[-1].get_as_image(), cmap='gray')
            # fig.set_tight_layout(True)
            # fig.canvas.draw()
            # fig.canvas.flush_events()
            # time.sleep(0.1)            
        print(fails)
        train_sets.append(new_image_set)
    #plt.show()
    return train_sets


def data_split(images: "list[Image]", split_factor):
    
    split_i = int(len(images) * split_factor)
    train_set, test_set = images[0:split_i], images[split_i:len(images)]

    return (train_set, test_set)


def data_loading(images_path: String, images_label_path: String):
    #training_images = open("mnist dataset\\training-images.txt")
    #training_labels = open("mnist dataset\\training-labels.txt")
    #validation_images = open("mnist dataset\\validation-images.txt")
    images: "list[Image]" = []

    with open(images_path) as images_data:
        for i in range(0, 2): #Ignore first 2 lines
            images_data.readline()
        [numi, rows, cols, digi] = images_data.readline().split() #read-metadata
        [numi, rows, cols] = [int(numi), int(rows), int(cols)] 
        digits = [int(x) for x in digi]

        try: #Add labels to Images
            images_labels = open(images_label_path) 
            for i in range(0, 2): #Ignore first 2 lines
                images_labels.readline()
            [t_numl, t_digl] = images_labels.readline().split() #read-metadata

            for _ in range(0, numi):
                images.append(Image.from_file(images_data, images_labels))
        except: #Images without labels
            for _ in range(0, numi):
                images.append(Image.from_file(images_data, None))

        random.shuffle(images)

    return (images, rows, cols, digits)

    


    #traning_images = open(sys.argv[1])    
    #traning_labels = open(sys.argv[2])
    #validation_images = open(sys.argv[3])

    
    
    
    #[v_numi, v_rowi, v_coli, v_digi] = validation_images.readline().split()
    
    

    
