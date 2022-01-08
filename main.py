import cv2
import numpy as np
import matplotlib.pyplot as plt
from wavepy import surface_from_grad
from math import pi
from colorsys import hls_to_rgb

things = ["cat", "frog", "hippo", "lizard", "pig", "scholar", "turtle"]

def obtainData(i):
    """
    :param i: the index of the thing in things list we want to obtain data for
    :return: a list of lightsource directions, a list of masked images lit with different lightsources
    """
    # Read all images
    paths = (
        fr".\PSData\PSData\{things[i]}\Objects\Image_01.png",
        fr".\PSData\PSData\{things[i]}\Objects\Image_02.png",
        fr".\PSData\PSData\{things[i]}\Objects\Image_03.png",
        fr".\PSData\PSData\{things[i]}\Objects\Image_04.png",
        fr".\PSData\PSData\{things[i]}\Objects\Image_05.png"
    )

    imgs = [cv2.imread(p,0) for p in paths]

    # Shadow area does not contribute to finding normal
    # Create masks
    threshold = 10

    def mask(img, thresh):
        mask = np.zeros_like(img)
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                if img[x][y] > thresh:
                    mask[x][y] = 255
                else:
                    mask[x][y] = 0
        return mask

    # Apply masks to images: cv2.bitwise
    #all images stores in imglist
    imglist = [cv2.bitwise_or(img, img, mask=mask(img, threshold)) for img in imgs]

    # obtain lightsource directions
    with open(fr".\PSData\PSData\{things[i]}\refined_light.txt") as file:
        sources1 = []
        for line in file:
            line = line.split()
            for x in line:
                sources1.append(float(x.replace(',', "")))

    sources1 = np.reshape(np.asarray(sources1), (20, 3))

    return sources1, imglist

sources, imglist = obtainData(0)

# Obtain normals
rows, cols = imglist[0].shape
def find_NormAlbedo(sources, imglist, rows, cols):
    '''
    :param sources: a list of light source coordinates as [x,y,z] coordinates per light source
                    (shape (20,3) for 20 sources)
    :param imglist: a list of all images for one object
    :param rows: shape[0] of every image
    :param cols: shape[1] of every image
    :return: returns normals and albedo's for an object
    '''
    normal = np.zeros_like(imglist[0], dtype=np.ndarray)
    albedo = np.zeros_like(imglist[0])

    #Build S (lightsources)
    S = []
    for i in range(len(imglist)):
        S.append(sources[i])

    # for every pixel: find I, compute normal and albedo
    for x in range(rows):
        for y in range(cols):
            I = []  # intensity matrix of pixel x,y per image
            for i in range(len(imglist)):
                img = imglist[i]
                I.append(img[x][y])

            # Least squares solution if S is invertible
            # pseudoinverse
            pseudoS = np.linalg.pinv(S)

            ntilde = pseudoS @ I

            p = np.linalg.norm(ntilde, ord=2)
            if p != 0.:
                n = ntilde / p
                n = n.flatten()
                # print(n)
                # print(n.shape)
            else:
                n = np.zeros_like(ntilde.flatten())

            normal[x][y] = n
            albedo[x,y] = p/8 #Values range from 0 to 1998.34. When tries to store in a byte, we only store the lower 8 bits

    return normal, albedo


normal, albedo = find_NormAlbedo(sources, imglist, rows, cols)
# print(normal.shape)
# print(albedo.shape)


# Estimate depth map from normals: Frankot Chellappa algorithm
def depthfromgradient(normalmap):
    '''
    :param normalmap: Previously obtained normals per pixel
    :return: Surface/Depth map from normalmap
    '''
    surfacep = np.zeros_like(normalmap)
    surfaceq = np.zeros_like(normalmap)
    for row in range(rows):
        for x in range(cols):
            #print(x)
            a, b, c = normalmap[row][x]
            #print(a, b, c)
            if c !=0:
                p = -a / c  # p=dZ/dx
                q = -b / c  # q=dZ/dy
                surfacep[row][x] = p
                surfaceq[row][x] = q
    return surface_from_grad.frankotchellappa(surfacep, surfaceq, reflec_pad=True)


surface = depthfromgradient(normal)

#cv2 need np array to visualize
surface = np.array(surface)
albedo = np.array(albedo)
normal = np.array(normal)


print(f'example of pixel in normalmap: {normal[0][0]}')
print(f'example of pixel in albedomap:{albedo[0][0]}')
print(f'example of pixel in normalmap:{surface[0][0]}')

#Visualize albedo
cv2.imshow('Albedo map', albedo)
cv2.waitKey()

#Surface map is a complex number and can be visualized by colorizing it
#Visualizing/colorizing comlpex array (def colorize(z)) adapted from https://stackoverflow.com/questions/17044052/matplotlib-imshow-complex-2d-array
def colorize(z):
    r = np.abs(z)
    arg = np.angle(z)

    h = (arg + pi)  / (2 * pi) + 0.5
    l = 1.0 - 1.0/(1.0 + r**0.3)
    s = 0.8

    c = np.vectorize(hls_to_rgb)(h,l,s) # --> tuples
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0,2)
    return c

plt.imshow(colorize(surface))
plt.show()

