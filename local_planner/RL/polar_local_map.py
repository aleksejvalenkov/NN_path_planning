import cv2
import numpy as np
import time

def cortasian_to_polar(image, robot_center, target, window, draw = False):
    # Convert robot_center and target from float to int
    robot_center = (int(robot_center[0]), int(robot_center[1]), np.rad2deg(robot_center[2]))
    target = (int(target[0]), int(target[1]))
    # Draw the target point on the image as a green circle
    image = cv2.circle(image, target, radius=10, color=(0, 255, 0), thickness=-1)
    
    # Change all white borders to red
    image[np.all(image >= [200, 200, 200], axis=-1)] = [0, 0, 255]

    # Extend the image by padding zeros around it
    padding = (window[1] // 2, window[0] // 2)
    image = cv2.copyMakeBorder(
        image,
        padding[1], padding[1], padding[0], padding[0],
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0]  # Padding with black (zeros)
    )
    robot_center = (robot_center[0] + padding[0], robot_center[1] + padding[1], robot_center[2])
    local_map = image[robot_center[1]-window[1]//2:robot_center[1]+window[1]//2,
                    robot_center[0]-window[0]//2:robot_center[0]+window[0]//2,
                    0:3]
    
    # Rotate the local map based on the robot's orientation (theta)
    rotation_matrix = cv2.getRotationMatrix2D((window[0] // 2, window[1] // 2), robot_center[2], 1.0)
    local_map = cv2.warpAffine(local_map, rotation_matrix, (window[0], window[1]))

    # Transform the local map from Cartesian to polar coordinates
    polar_local_map = cv2.linearPolar(
        local_map,
        (window[1] // 2, window[0] // 2),  # Center of the transformation (robot's position in the local map)
        (max(window)*2**0.5)//2,                  # Maximum radius for the transformation
        cv2.WARP_FILL_OUTLIERS             # Fill outliers outside the bounds
    )
    # out = np.hstack((local_map, polar_local_map))

        # resize is undefined; if resizing is needed, use cv2.resize or remove this line
    polar_local_map = cv2.resize(polar_local_map, (100, 100), interpolation=cv2.INTER_LINEAR)

    if draw:
        cv2.imshow("Loaded Image", polar_local_map)
        cv2.waitKey(1)

    return polar_local_map

def example():
    # Read the image
    image_path = "/home/alex/Documents/NN_path_planning/global_planner/map/map_bin_ext.jpg"  # Replace with your image path
    image = cv2.imread(image_path)

    # Check if the image was successfully loaded
    if image is None:
        print("Error: Unable to load image. Check the file path.")
    else:
        print("Image loaded successfully.")

    print('image shape = ', image.shape)
    target = (200,300)
    robot_center = (350,200)
    window = (400,400)
    for i in range(1600):
        robot_center = (i,200)
        start_time = time.time()
        polar = cortasian_to_polar(image, robot_center, target, window)
        end_time = time.time()
        print(f"Processing time for robot_center {robot_center}: {end_time - start_time:.6f} seconds")

            # Display the image (optional)
        cv2.imshow("Loaded Image", polar)
        cv2.waitKey(1)
        # cv2.destroyAllWindows()

# example()

