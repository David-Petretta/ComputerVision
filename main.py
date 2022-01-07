import cv2
import numpy as np
import matplotlib.pyplot as plt
from wavepy import surface_from_grad

things = ["cat", "frog", "hippo", "lizard", "pig", "scholar", "turtle"]


def obtainData(i):
    """
    :param i: the index of the thing in things list we want to obtain data for
    :return: a list of lightsource directions, a list of masked images lit with different lightsources
    """
    # Read all images
    path1 = fr".\PSData\PSData\{things[i]}\Objects\Image_01.png"
    path2 = fr".\PSData\PSData\{things[i]}\Objects\Image_02.png"
    path3 = fr".\PSData\PSData\{things[i]}\Objects\Image_03.png"
    path4 = fr".\PSData\PSData\{things[i]}\Objects\Image_04.png"
    path5 = fr".\PSData\PSData\{things[i]}\Objects\Image_05.png"

    img1 = cv2.imread(path1, 0)  # flag 0 reads in grayscale
    img2 = cv2.imread(path2, 0)
    img3 = cv2.imread(path3, 0)
    img4 = cv2.imread(path4, 0)
    img5 = cv2.imread(path5, 0)

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
    img1m = cv2.bitwise_or(img1, img1, mask=mask(img1, threshold))
    img2m = cv2.bitwise_or(img2, img2, mask=mask(img2, threshold))
    img3m = cv2.bitwise_or(img3, img3, mask=mask(img3, threshold))
    img4m = cv2.bitwise_or(img4, img4, mask=mask(img4, threshold))
    img5m = cv2.bitwise_or(img5, img5, mask=mask(img5, threshold))

    imglist = [img1m, img2m, img3m, img4m, img5m]

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
            albedo[x][y] = p

    return normal, albedo


normal, albedo = find_NormAlbedo(sources, imglist, rows, cols)
print(type(normal[70]))
print(type(albedo[70]))
print(type(normal[70][0]))
print(type(albedo[70][0]))
plt.imshow(albedo)
plt.title("Albedo")
#plt.show()

print(normal.shape)
print(albedo.shape)
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
print(surface.shape)

#cv2 need np array to visualize
surface = np.array(surface)
albedo = np.array(albedo)
normal = np.array(normal)

print(surface[0][0])

cv2.imshow('Albedo', albedo)
cv2.waitKey()

cv2.imshow('DepthMap', surface)
cv2.waitKey()
