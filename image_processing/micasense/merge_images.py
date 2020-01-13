import cv2
import numpy as np
from PIL import Image

##############################################################
### FUNCTIONS
##############################################################

# 1. Extract ORB keypoints and descriptors from a gray image


def extract_features(gray):

    orb = cv2.ORB_create()
    kp, desc = orb.detectAndCompute(gray, None)

    return kp, desc


# 2. Find corresponding features between the images

def find_matches(kp1, desc1, kp2, desc2):

    # create BFMatcher object (for ORB features)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(desc1, desc2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # convert first 10 matches from KeyPoint objects to NumPy arrays
    points1 = np.float32([kp1[m.queryIdx].pt for m in matches[0:15]])
    points2 = np.float32([kp2[m.trainIdx].pt for m in matches[0:15]])

    return points1, points2


# 3. Find homography between the points

def find_homography(points1, points2):

    # convert the keypoints from KeyPoint objects to NumPy arrays
    src_pts = points2.reshape(-1, 1, 2)
    dst_pts = points1.reshape(-1, 1, 2)

    # find homography
    homography, mask = cv2.findHomography(src_pts, dst_pts)

    return homography


# 4.1 Calculate the size and offset of the stitched panorama

def calculate_size(img1, img2, homography):

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners2 = np.float32([[[0, 0], [0, h2 - 1], [w2 - 1, h2 - 1], [w2 - 1, 0]]])

    # remap the coordinates of the projected image onto the panorama image space
    transformedCorners2 = cv2.perspectiveTransform(corners2, homography)
    #print(transformedCorners2)

    #offset = (transformedCorners2[0][0][0], transformedCorners2[0][0][1])

    offset = (0, np.abs(transformedCorners2[0][3,1]))

    size = (np.ceil(transformedCorners2[0][3,0]), np.ceil(transformedCorners2[0][2,1]-transformedCorners2[0][3,1]))

    homography[0:2, 2] += offset
    #print(offset)
    #print(size)

    return size, offset


# 4.2 Combine images into a panorama
def merge_images(image1, image2, homography, size, offset):

    size = (image1.shape[1], image1.shape[0])

    dst = cv2.warpPerspective(image2, homography, size)

    #dst[int(offset[1]): (image1.shape[0] + int(offset[1])), 0:image1.shape[1]] = image1

    # left = 0
    # top = 0
    # bottom = image1.shape[0]
    # right = image1.shape[1]

    #panorama = dst.crop((left, top, right, bottom))
    panorama = dst
    return panorama

def contrast_stretch(im):
    """
    Performs a simple contrast stretch of the given image, from 5-95%.
    """
    in_min = np.percentile(im, 5)
    in_max = np.percentile(im, 95)

    out_min = 0.0
    out_max = 255.0

    out = im - in_min
    out *= ((out_min - out_max) / (in_min - in_max))
    out += in_min

    return out

### --- No need to change anything below this point ---

### Connects corresponding features in the two images using yellow lines
def draw_matches(image1, image2, points1, points2):

  # Put images side-by-side into 'image'
    (h1, w1) = image1.shape[:2]
    (h2, w2) = image2.shape[:2]
    image = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    image[:h1, :w1] = image1
    image[:h2, w1:w1 + w2] = image2

  # Draw yellow lines connecting corresponding features.
    for (x1, y1), (x2, y2) in zip(np.int32(points1), np.int32(points2)):
        cv2.line(image, (x1, y1), (x2 + w1, y2), (0, 255, 255))

    return image


##############################################################
### MAIN PROGRAM
##############################################################

# Load images
img1 = cv2.imread('IMG_0047_2.tif')
img2 = cv2.imread('IMG_0047_5.tif')

# Convert images to grayscale (for ORB detector).
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 1. Detect features and compute descriptors.

kp1, desc1 = extract_features(gray1)
kp2, desc2 = extract_features(gray2)
#print('{0} features detected in image1').format(len(kp1))
#print('{0} features detected in image2').format(len(kp2))

orb1 = cv2.drawKeypoints(gray1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
orb2 = cv2.drawKeypoints(gray2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('Image1_orf.JPG', orb1)
cv2.imwrite('Image2_orb.JPG', orb2)
#cv2.imshow('Features 1', orb1)
#cv2.imshow('Features 2', orb2)
#cv2.waitKey(0)

# 2. Find corresponding features

points1, points2 = find_matches(kp1, desc1, kp2, desc2)
#print('{0} features matched').format(len(points1))

match = draw_matches(img1, img2, points1, points2)
cv2.imwrite('matching.JPG', match)
#cv2.imshow('Matching', match)
#cv2.waitKey(0)

# 3. Find homgraphy

H = find_homography(points1, points2)
#print(H)
# 4. Combine images into a panorama

(size, offset) = calculate_size(img1, img2, H)

#print('output size: {0}  offset: {1}').format(size, offset)

panorama = merge_images(img1, img2, H, size, offset)
cv2.imwrite("panorama.jpg", panorama)
#cv2.imshow('Panorama', panorama)

nir = panorama.astype(np.uint8)
nir[np.isnan(nir)] = 0
red = img1.astype(np.uint8)

cv2.imshow('nir', nir)
cv2.imshow('red', red)

nir = panorama.astype(np.uint8) + 0.0000000000001
nir[np.isnan(nir)] = 0
red = img1.astype(np.uint8) + 0.000000000001

np.seterr(divide='ignore', invalid='ignore')
ndvi = ((nir - red) / (nir + red)).astype(float)

ndvi[np.isnan(ndvi)] = 0

ndvi_values = np.count_nonzero(ndvi > 0.2)
total_values = ndvi.shape[0]*ndvi.shape[1]
percent = np.round((ndvi_values / total_values) * 100, 2)

ndvi[ndvi < 0.1] = 0

# kernel = np.ones((2, 2), np.uint8)
# opening = cv2.morphologyEx(ndvi, cv2.MORPH_OPEN, kernel)
# ndvi = opening

ndvi_new = contrast_stretch(ndvi).astype(np.uint8)

ndvi_final = cv2.applyColorMap(ndvi_new, cv2.COLORMAP_JET)


print('Max ndvi value', np.amax(ndvi))
print('Min ndvi value', np.amin(ndvi))
print('ndvi values', ndvi_values)
print('size', total_values)
print('percentage of vegetation', percent)

cv2.imshow('ndvi', ndvi)
cv2.imshow('ndvi final', ndvi_final)


cv2.waitKey(0)
cv2.destroyAllWindows()
