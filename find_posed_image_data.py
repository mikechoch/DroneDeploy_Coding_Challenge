"""
Michael DiCioccio
DroneDeploy Coding Challenge Option 1
find_posed_image_data.py
"""

# Libs
import sys
import cv2
import math
import numpy
from pathlib import Path
from matplotlib import pyplot as plt

# Globals
VALID_NORM = 1e-6
FOCAL_WIDTH = 330
FOCAL_HEIGHT = 330


def isRotationMatrix(R):
    """
    Verifies that a rotation matrix is a valid rotation matrix
    Compares the transpose rotation matrix to the normal one by taking the dot product
    Finds the normalized value of the dot product and if it is less than the VALID_NORM it is a valid rotation matrix
    :return: True or False
    """
    Rt = numpy.transpose(R)
    expected_identity = numpy.dot(Rt, R)
    identity = numpy.identity(3, dtype=R.dtype)
    norm_val = numpy.linalg.norm(identity - expected_identity)
    return norm_val < VALID_NORM


def rotationMatrixToEulerAngles(R):
    """
    Takes in a rotation matrix and calculates the x, y, and z euler angles
    :return: an array of x, y, and z euler angles
    """
    if isRotationMatrix(R):
        m00 = R[0][0]
        m01 = R[0][1]
        m02 = R[0][2]
        m10 = R[1][0]
        m11 = R[1][1]
        m12 = R[1][2]
        m20 = R[2][0]
        m21 = R[2][1]
        m22 = R[2][2]

        x_theta = math.atan2(m12, m22)
        c2 = math.sqrt(m00**2 + m01**2)
        y_theta = math.atan2(-m02, c2)
        s1 = math.sin(x_theta)
        c1 = math.cos(x_theta)
        z_theta = math.atan2(s1 * m20 - c1 * m10, c1 * m11 - s1 * m21)

        return [x_theta, y_theta, z_theta]

    return


def findDescriptorMatches(des1, des2):
    """
    Takes in two sets of descriptor arrays and uses a FlannBasedMatcher to find the best matches between images
    :return: an array of best arrays
    """
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    return matches


def getGoodMatchDescriptors(matches):
    """
    Takes matches of descriptors between two images and finds ones that meet the threshold distance check
    :return: an array of the good matches
    """
    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append([m])

    return good


def getImagePoints(kp1, kp2):
    """
    Get the points of the pattern and points of the desired image
    :return: points for the pattern and points for the desired image
    """
    pattern_pts = []
    desired_img_pts = []
    n = min(len(kp1), len(kp2))
    for i in range(n):
        pattern_pts.append(kp1[i].pt)
        desired_img_pts.append(kp2[i].pt)

    return numpy.float32(pattern_pts), numpy.float32(desired_img_pts)


def getPoseMatrix(src_pts, dst_pts, desired_img):
    """
    Finds a few matrices using the src points (pattern) and dst points (desired image) to find the pose matrix
    The pose matrix has a few different data types in it
    It includes the scalar constant for the image, rotation matrix, translation matrix, color matrix, etc.
    :return: the constant, rotation matrix, and translation matrix
    """
    # Obtain the camera matrix for the desired img
    camera_mat = getCameraMat(desired_img)

    # Obtain the essential matrix, which will be used to get the estimated pose matrix
    essential_mat, essential_mat_mask = cv2.findEssentialMat(src_pts, dst_pts, camera_mat)
    printMatrix("Essential Matrix", essential_mat)

    # Obtain the estimated pose by using the essential matrix (similar to the homography matrix)
    pose_mat = cv2.recoverPose(essential_mat, src_pts, dst_pts, camera_mat)
    # printMatrix("Pose Data", pose_mat)

    return pose_mat[:3]


def getCameraMat(img):
    """
    Creates a camera matrix using the focal width, focal height, center focus x, and center focus y
    :return: camera matrix based off the input image
    """
    width, height = img.shape[:3]

    fx = FOCAL_WIDTH
    fy = FOCAL_HEIGHT
    cx = width / 2
    cy = height / 2

    return numpy.float32([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def printTranslations(title, trans_mat, c):
    """
    Calculates and prints the X, Y, and Z translations
    """
    print("\n" + title + ":")
    print("X-Translation: " + str(c * trans_mat[2]))
    print("Y-Translation: " + str(c * trans_mat[1]))
    print("Z-Translation: " + str(c * trans_mat[0]))


def printAxisRotations(title, euler_mat):
    """
    Prints the X, Y, and Z axis rotations in degrees
    """
    print("\n" + title + ":")
    print("X-Axis Rotation: " + str(euler_mat[2] * (180 / math.pi)))
    print("Y-Axis Rotation: " + str(euler_mat[1] * (180 / math.pi)))
    print("Z-Axis Rotation: " + str(euler_mat[0] * (180 / math.pi)))


def printMatrix(title, mat):
    """
    Prints a matrix with a title
    """
    print("\n" + title + ":")
    print(mat)


def main():
    # Based on entered image path, it creates the pattern path using split and join
    desired_img_path = ""
    while not Path(desired_img_path).is_file():
        desired_img_path = str(input("Enter the image file path to find the camera pose for: "))

    # Splits the desired img path and uses it to create the pattern.png path (both images should be in same directory)
    desired_img_list = desired_img_path.split('/')
    pattern_path_png = '/'.join(desired_img_list[:-1]) + "/pattern.png"
    if not Path(pattern_path_png).is_file():
        print("Could not find pattern.png inside of the path to the desired image.")
        return

    # Convert pattern.png into a jpg if it doesn't exist
    # pattern_path_jpg = pattern_path_png[:-3] + "jpg"
    # if not Path(pattern_path_jpg).is_file():
    #     # Grabs the pattern.png and then write a new file converting it into a jpg
    #     pattern_img = cv2.imread(pattern_path_png)
    #     cv2.imwrite(pattern_path_jpg, pattern_img)

    # Read in the new pattern.jpg (or pattern.png) and the desired image
    # pattern_img = cv2.imread(pattern_path_jpg, 0)
    pattern_img = cv2.imread(pattern_path_png, 0)
    desired_img = cv2.imread(desired_img_path, 0)

    # Print message start
    print("Please wait, analyzing images...")

    # Had a few options for finding KeyPoints between the pattern and the desired image
    # Decided to use sift because it seemed a bit more accurate when getting KeyPoints from different orientations
    # orb = cv2.ORB_create()
    # surf = cv2.xfeatures2d.SURF_create()
    sift = cv2.xfeatures2d.SIFT_create()

    # Obtain the KeyPoints and Descriptors for the pattern and desired image
    pattern_kp, pattern_des = sift.detectAndCompute(pattern_img, None)
    desired_kp, desired_des = sift.detectAndCompute(desired_img, None)

    # Get an array of matches between descriptors in both images and get good matches based off a threshold distance
    matches = findDescriptorMatches(pattern_des, desired_des)
    good_matches = getGoodMatchDescriptors(matches)

    # Get the points of the pattern and points of the desired image
    pattern_pts, desired_img_pts = getImagePoints(pattern_kp, desired_kp)

    # Obtain the image constant c, rotation matrix, and translation matrix from the estimated pose
    c, rotation_mat, translation_mat = getPoseMatrix(pattern_pts, desired_img_pts, desired_img)
    print("\nScalar: " + str(c))
    printMatrix("Rotation Matrix", rotation_mat)
    printMatrix("Translation Matrix", translation_mat)

    # Calculate the Euler Angles in radians based off of the rotation matrix
    euler_angles_mat = rotationMatrixToEulerAngles(rotation_mat)
    printAxisRotations("Axis Rotations (Degrees -180˚ to 180˚)", euler_angles_mat)

    # Calculate the coordinates of the image based off the translation matrix
    printTranslations("Translations (X, Y, Z)", translation_mat[:, 0], c)

    # Print message end
    print("\nComplete.")

    # Create the KeyPoint connected image (pattern and desired image) then plot it to display it
    pattern_connected_img = cv2.drawMatchesKnn(pattern_img,
                                               pattern_kp,
                                               desired_img,
                                               desired_kp,
                                               good_matches,
                                               None,
                                               flags=2)

    plt.imshow(pattern_connected_img), plt.show()

    return


if __name__ == "__main__":
    main()
