import cv2
import os
import re
import numpy as np
from tqdm import tqdm

## Script to run run from within datasets/gtri folder


def split_video_to_frames(video_path, output_folder):
    # Check if the output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Capture the video
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_count = 0
    while True:
        # Read the next frame from the video
        ret, frame = cap.read()

        # If there are no more frames, break out of the loop
        if not ret:
            break

        # Construct the output file name
        output_filename = os.path.join(output_folder, f"frame_{frame_count:06d}.png")

        # Save the frame as a PNG file
        cv2.imwrite(output_filename, frame)

        # Print status
        print(f"Saved {output_filename}")

        # Increment the frame count
        frame_count += 1

    # Release the video capture object
    cap.release()
    print("Done!")

# # Example usage
# video_path = 'VID_012.MOV'
# output_folder = 'ae_engine/'
# split_video_to_frames(video_path, output_folder)

# write traj file
def write_rgb():
    with open("rgb.txt", 'a') as f:
        f.writelines([
            "# color images\n",
            "# file: \'VID_012.mov\'\n",
            "# timestamp filename\n"
        ])
        for img_fname in os.listdir("rgb/"):
            frame_num = re.findall(r'\d+', img_fname)[0]
            img_fp = os.path.join("rgb", img_fname)
            f.write(f"{frame_num} {img_fp}\n")


def aruco_display(corners, ids, rejected, image):
	if len(corners) > 0:
		# flatten the ArUco IDs list
		ids = ids.flatten()
		# loop over the detected ArUCo corners
		for (markerCorner, markerID) in zip(corners, ids):
			# extract the marker corners (which are always returned in
			# top-left, top-right, bottom-right, and bottom-left order)
			corners = markerCorner.reshape((4, 2))
			(topLeft, topRight, bottomRight, bottomLeft) = corners
			# convert each of the (x, y)-coordinate pairs to integers
			topRight = (int(topRight[0]), int(topRight[1]))
			bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
			bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
			topLeft = (int(topLeft[0]), int(topLeft[1]))

			cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
			cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
			cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
			cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
			# compute and draw the center (x, y)-coordinates of the ArUco
			# marker
			cX = int((topLeft[0] + bottomRight[0]) / 2.0)
			cY = int((topLeft[1] + bottomRight[1]) / 2.0)
			cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
			# draw the ArUco marker ID on the image
			cv2.putText(image, str(markerID),(topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
				0.5, (0, 255, 0), 2)
			print("[Inference] ArUco marker ID: {}".format(markerID))
			# show the output image
	return image


def rotation_matrix_to_quaternion(R):
    """
    Convert a rotation matrix to a quaternion.
    The rotation matrix is assumed to be in the form of a 3x3 array.
    The quaternion is returned as a numpy array [w, x, y, z].
    """
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]

    tr = m00 + m11 + m22

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2  # S=4*qw
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2  # S=4*qx
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2  # S=4*qy
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2  # S=4*qz
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S

    return qx, qy, qz, qw


def getIdFromFrameNum(num):
    if num > 1150 and num < 1415:
        return 1
    elif (num > 1970 and num < 2230):
        return 2
    elif num > 3450 and num < 3720:
        return 3
    elif num > 4800 and num < 5050:
        return 4
    elif num > 5650 and num < 5765:
        return 5
    # marker with "5" on it skipped in video (6th marker)
    # marker with 6 on it has height but dont know y value so skipping for now
    else:
        return -1


def find_aruco_poses():
    f = open('groundtruth.txt', 'a')
    f.writelines([
        "# ground truth trajectory\n",
        "# file: \'VID_012.mov\'\n",
        "# timestamp tx ty tz qx qy qz qw\n"
    ])
    for i, img_fname in tqdm(enumerate(sorted(os.listdir('rgb/')))):
        if i < 1150:
            continue

        img_fp = os.path.join('rgb', img_fname)
        image = cv2.imread(img_fp)

        # load the ArUCo dictionary, grab the ArUCo parameters, and detect
        # the markers
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        parameters =  cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)

        markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(image)
        
        # detected_markers = aruco_display(markerCorners, markerIds, rejectedCandidates, image)

        distortion_coefficients = np.zeros((5,))
        intrinsic_mat = np.zeros((3,3))
        fx= 65.0
        fy= 65.0
        cx= 640.0
        cy= 360.0
        intrinsic_mat[0, 0] = fx
        intrinsic_mat[1,1] = fy
        intrinsic_mat[0,2] = cx
        intrinsic_mat[1,2] = cy
        marker_size = 20.0
        if len(markerCorners) > 0: # only take first detected marker for reference

            id = getIdFromFrameNum(i)
            if id==-1:
                 continue

            marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                                [marker_size / 2, marker_size / 2, 0],
                                [marker_size / 2, -marker_size / 2, 0],
                                [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)

            c = markerCorners[0]
            _, rvec, tvec = cv2.solvePnP(marker_points, c, intrinsic_mat, distortion_coefficients, False, cv2.SOLVEPNP_IPPE_SQUARE)
            # x=-x and z=-z to switch to correct coord system
            rvec[0] *= -1
            rvec[2] *= -1
            tvec[0] *= -1
            tvec[2] *= -1 # do all of these inversions work? they should still be approx correct
            # then convert rvec to R mat, 
            Rmat = cv2.Rodrigues(rvec)[0]
            # then invert to find w2c
            w2c_R = Rmat.T
            w2c_t = -Rmat.T @ tvec
            # then save in groundtruth.txt
            # transform by 85mm in z and 305 in y
            w2c_t[1] += 304.8 * id
            w2c_t[2] += 85.85

            # convert to quaternion and write to file
            qx, qy, qz, qw = rotation_matrix_to_quaternion(w2c_R)
            tx, ty, tz = list(w2c_t)
            frame_num = re.findall(r'\d+', img_fname)[0]
            f.write(f"{frame_num} {tx[0]} {ty[0]} {tz[0]} {qx} {qy} {qz} {qw}\n") #TODO: not sure if w is in the right place
    f.close()



find_aruco_poses()