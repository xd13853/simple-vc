import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

IMPATH = 'visual_cortex_testim.png'

EPSILON = 1e-9

OUTPUT_IMAGE_SIZE = (40,30)
W_out, H_out = OUTPUT_IMAGE_SIZE

INPUT_IMAGE_SIZE = (40,30)
W_in, H_in = INPUT_IMAGE_SIZE

im = cv2.imread(IMPATH)
im = cv2.resize(im, INPUT_IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)
H,W,C = im.shape

# compute the largest possible value of radius - the length of the line segment connecting the
# centre of the image to the corners
RMAX = math.sqrt(((H/2.)**2) + ((W/2.)**2))
XMAX = math.log(RMAX)

def showims(img_array_list, label_list=None):
    fig = plt.figure()
    for i, img in enumerate(img_array_list):
        a = fig.add_subplot(1, len(img_array_list), i+1)
        imgplot = plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if label_list is not None:
            a.set_title(label_list[i])
    plt.show()

def move_origin_topleft_2_centre(point, im):
    '''
    Since image frame has origin in top left, translate the origin to the
    centre of the image.
    '''
    h,w,c = im.shape
    x,y = point
    x -= w/2
    y -= h/2
    y *= -1
    return (x,y)

def move_origin_centre_2_topleft(point, im):
    '''
    The inverse of move_origin_topleft_2_centre()
    '''
    h,w,c = im.shape
    x,y = point
    x += w/2
    y += h/2
    # y *= -1
    return (x,y)

def cartesian_2_polar(point):
    '''
    Transform cartesian image coords (with origin in the centre of the image) into
    polar coordinates.
    '''
    x,y = point
    r = math.sqrt(x**2 + y**2)
    theta = math.atan2(y,x)
    return (r,theta)

def polar_2_cortex(point):
    '''
    The (simplified) mapping as described in
    https://plus.maths.org/content/uncoiling-spiral-maths-and-hallucinations
    '''
    r,theta = point
    x = math.log(r + EPSILON)
    y = theta
    return (x,y)

def output_2_cortex_rescale(point, im_output):
    '''
    The output image has origin at the topleft corner. This method converts points
    from this output image frame into the frame of the visual cortex, which is
    cartsian with origin at the centre. Its y-axis ranges from pi to -pi, and
    its x-axis ranges from 0 to XMAX.
    '''
    # translate origin from topleft to centre
    x,y = move_origin_topleft_2_centre(point, im_output)
    print('output point moved to recentre origin at centre {}'.format((x,y)))
    # rescale axes
    l = (XMAX * x) / W_out
    m = (2*math.pi* (y / H_out)) - math.pi
    return (l,m)

def cortex_2_retina(point):
    '''
    The inverse of polar_2_cortex()
    '''
    x,y = point
    theta = y
    r = math.exp(x)
    return (r, theta)

def polar_2_cartesian(point):
    r, theta = point
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return (x, y)

def dst_2_src(point_output, im_input, im_output):
    '''
    Does the whole pipeline of transformations from output to input images
    '''
    print('====== {} ======'.format(point_output))
    point_cortex = output_2_cortex_rescale(point_output, im_output)
    print('Moved and rescaled output coord to cortex coord: {}'.format(point_cortex))
    point_retina = cortex_2_retina(point_cortex)
    print('Mapped cortex coord to retina coord: {}'.format(point_retina))
    point_cartesian = polar_2_cartesian(point_retina)
    print('Mapped retina coord to cartesian coord: {}'.format(point_cartesian))
    point_input = move_origin_centre_2_topleft(point_cartesian, im_input)
    print('Moved origin from centre to topleft: {}'.format(point_input))
    return point_input


# create the mappings
lookup_table = np.zeros(shape=(OUTPUT_IMAGE_SIZE[1], OUTPUT_IMAGE_SIZE[0], 2))
# iterate over output image width (rows):
for i in range(OUTPUT_IMAGE_SIZE[0]):
    # iterate over output image height (cols):
    for j in range(OUTPUT_IMAGE_SIZE[1]):
        dst_coord = (i, j)
        # find the location in the source image that should be sampled from
        src_coord = dst_2_src(dst_coord, im, lookup_table)
        x_in, y_in = src_coord
        lookup_table[j][i] = np.array([x_in, y_in])
        # outim_like[j][i] = np.array([i, j]) # sanity check to perform identity mapping

map_x = lookup_table[:, :, 0].astype(np.float32)
map_y = lookup_table[:, :, 1].astype(np.float32)

outim = cv2.remap(im, map_x, map_y, interpolation=cv2.INTER_CUBIC)
showims([im, outim], ['input', 'output'])