import cv2
import math
import numpy as np
import datetime
import requests
import os
import logging
from logging.handlers import RotatingFileHandler
import signal
import sys
# import tower_light as tl
import threading

class ImageCapture:
    def __init__(self, image_path):
        self.image_path = image_path
        self.frame = cv2.imread(image_path)
        self._opened = self.frame is not None
        if not self._opened:
            print(f"Error: Unable to load image at {image_path}")
        else:
            pass

    def isOpened(self):
        return self._opened

    def read(self):
        if not self._opened or self.frame is None:
            return False, None
        # คืนสำเนาเฟรมเดิมทุกครั้ง เพื่อให้แก้ไขภาพในลูปได้โดยไม่ทับต้นฉบับ
        return True, self.frame.copy()

    def release(self):
        self._opened = False


# Set up paths for various files and directories used in the program
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the absolute path of the script's directory
running_number_path = os.path.join(script_dir, 'running_number.txt')  # Path to store the running number
calibration_data_path = os.path.join(script_dir, 'calibration_data/calibration_data_7x5_3040p.npz')  # Path to the camera calibration data
img_output_path = os.path.join(script_dir, 'output')  # Directory where output images will be saved
log_dir = os.path.join(script_dir, 'app.log')  # Path for log file

# สร้างโฟลเดอร์ output หากยังไม่มี (ไม่กระทบโครงสร้าง process)
os.makedirs(img_output_path, exist_ok=True)

# Initialize towerlight
# tl.initialize()

# Configure logging
logger = logging.getLogger('MTL_MEASUREMENT')
logger.setLevel(logging.DEBUG)  # Set the logging level to DEBUG

# Create a handler for rotating log files to avoid large log files
handler = RotatingFileHandler(log_dir, maxBytes=10*1024*1024, backupCount=5, mode='w')
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))

# Add the handler to the logger
logger.addHandler(handler)

# Flag to control whether to display windows with OpenCV, useful for headless environments
SHOW_WINDOWS = True

# Function to calculate the dimensions of a contour's bounding rectangle (approximation)
'''
def cal_dim(approx):
    dist_list = []
    c = 0
    # Calculate the Euclidean distance between each pair of points
    for i in range(len(approx)):
        for j in range(i+1, len(approx)):
            dist = math.sqrt(math.pow(approx[i][0][0] - approx[j][0][0], 2) + math.pow(approx[i][0][1] - approx[j][0][1], 2))
            dist_list.append(dist)
            c += 1
    dist_list = sorted(dist_list)[:4]  # Keep the four smallest distances
    height = int((dist_list[0] + dist_list[1]) / 2)  # Average of the smallest two for height
    width = int((dist_list[2] + dist_list[3]) / 2)  # Average of the next two for width

    return width, height 
'''

def cal_dim(contours, hierarchy, px2mm_x, px2mm_y):

    total_area_px = 0
    max_rect = None
    max_cnt = None

    for i, cnt in enumerate(contours):
        area_px = cv2.contourArea(cnt)
        if hierarchy[0][i][3] != -1:
            total_area_px -= area_px
        else: 
            total_area_px += area_px

            if max_cnt is None or area_px > cv2.contourArea(max_cnt):
                max_cnt = cnt
                max_rect = cv2.minAreaRect(cnt)

    area_mm2 = total_area_px * px2mm_x * px2mm_y

    (cx, cy), (w, h), angle = max_rect
    width_mm = w * px2mm_x
    height_mm = h * px2mm_y

    return {
        "area_mm2": area_mm2,
        "width_mm": width_mm,
        "height_mm": height_mm,
        "angle": angle,
        "contour": max_cnt
    }

# Function to rotate an image by a given angle
def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)  # Calculate the image center
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)  # Compute the rotation matrix
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)  # Apply the rotation
    return result

# Placeholder function for motion detection (currently not implemented)
def motion_detection(image):
    pass

# Function to get the next running number from a file, used to uniquely name images
def get_next_running_number(filename=running_number_path):
    logger.debug(f'Trying to read the running number from {filename}')
    try:
        # Read the current running number from the file
        with open(filename, "r") as file:
            running_number = int(file.read().strip()) + 1
    except FileNotFoundError:
        logger.error(f'File {filename} not found, starting with running number 1')
        running_number = 1  # Start with 1 if the file does not exist
        # thread_tl_r = threading.Thread(target=tl.turn_on_pin_b)
        # thread_tl_r.start()
    except ValueError:
        # Handle cases where the file is empty or contains invalid data
        logger.error(f'Invalid value in {filename}, starting with running number 1')
        running_number = 1
        # thread_tl_r = threading.Thread(target=tl.turn_on_pin_b)
        # thread_tl_r.start()
    except Exception as e:
        logger.error(f'Unexpected error: {e}, starting with running number 1')
        running_number = 1
        # thread_tl_r = threading.Thread(target=tl.turn_on_pin_b)
        # thread_tl_r.start()

    # Reset the counter if it exceeds 9999
    if running_number > 9999:
        running_number = 1

    # Update the file with the new running number
    try:
        with open(filename, "w") as file:
            file.write(str(running_number))
    except Exception as e:
        logger.error(f'Failed to write running number to file {filename}: {e}')
        # thread_tl_r = threading.Thread(target=tl.turn_on_pin_b)
        # thread_tl_r.start()

    # Return the running number formatted as a four-digit string with leading zeros
    return format(running_number, '04d')


def load_calibration_data(calibration_data_path):
    try:
        # Load the calibration data from the specified file
        calibration_data = np.load(calibration_data_path)
        logger.debug(f'Calibration data loaded successfully from {calibration_data_path}')
        
        # Extract camera matrix and distortion coefficients
        mtx, dist = calibration_data['mtx'], calibration_data['dist']
        return mtx, dist
    
    except Exception as e:
        # Log the error and exit the program if loading fails
        logger.error(f'Failed to load calibration data from {calibration_data_path}: {e}')
        logger.error('Exiting the program')
        # thread_tl_r = threading.Thread(target=tl.turn_on_pin_b)
        # thread_tl_r.start()
        exit()


def send_data_to_server(url, files, body, headers):

    return False  # Indicate that the request failed


def prepare_image_file(im_fname):
    return {'image': open(im_fname, 'rb')}


def prepare_request_body(running_number, object_area, object_width, object_height):
    body = {
        'material_id': f'mtal{running_number}', 
        'area': '{}'.format(round(object_area, 1)),
        'width': '{}'.format(round(object_width, 1)),
        'height': '{}'.format(round(object_height, 1)),
        'unit': 'mm'
    }
    return body


# Load the calibration data for camera undistortion
mtx, dist = load_calibration_data(calibration_data_path)

# Color range parameters for detecting the blue background
blue_Lower = (90, 120, 120)
# blue_Lower = (90, 10, 50)
# blue_Upper = (150, 255, 255)
blue_Upper = (150, 255, 255)

# Dimensions of the original object (in millimeters)
original_width = 615
original_height = 425

frame_number = 0  # Initialize the frame counter
motion_detected = False  # Flag to indicate if motion is being detected

# Constants for the program logic
WARMUP = 10  # Number of frames to skip at the beginning for camera warmup
THRESH = 10  # Threshold for motion detection
ASSIGN_VALUE = 255  # Value used in thresholding

# Server destination for uploading images and data
url = ""
headers = {
    #'authorization': "Bearer {token}"  # Example for authorization header if needed
}

image_path = os.path.join(script_dir, 'img/img2.jpg')  # แก้เป็นไฟล์รูปของคุณ

cap = ImageCapture(image_path)

if not cap.isOpened():
    print("Error: Unable to open the image source.")
    exit()

# Initialize kernel for glare reduction
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

while cap.isOpened():
    ret, frame = cap.read()
    
    frame_number += 1

    if ret and frame is not None:
        frame = cv2.resize(frame, (4032, 3040))
    
    if not ret or frame is None:
        logger.debug('Failed to capture video frame, exiting loop.')
        logger.error('Failed to capture video frame.')
        break  # Exit the loop if frame capture fails

    # Run logic only on the first frame to set up initial parameters
    if frame_number == 1:
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        frame_area = int(frame_height * frame_width)

        # Set pixel-to-mm conversion ratios
        pixel_mm_ratio_width = 0.5319  # Conversion ratio for width
        pixel_mm_ratio_height = 0.5284  # Conversion ratio for height

    frame_HD = frame.copy()
    frame = cv2.resize(frame, (1008,760))   # Resize for faster processing
    # frame = frame[80:600, 80:1000]   # Crop the undistorted image to a specific region

    # Convert the frame to grayscale and apply Gaussian blur for motion detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # Skip the first few frames for camera warmup
    if frame_number < WARMUP:
        if SHOW_WINDOWS:
            cv2.imshow("Raw Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        continue

    elif frame_number == WARMUP:
        # Capture the background frame for motion detection
        prev_frame = frame
        if SHOW_WINDOWS:
            cv2.imshow("Raw Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        continue

    key = cv2.waitKey(1)

    # Check if numpad key "1" is pressed, then start the measurement
    if key == ord('1'):

        # Undistort the captured frame using the calibration data
        frame_HD = cv2.undistort(frame_HD, mtx, dist, None, mtx)
        frame_HD = rotate_image(frame_HD,2.4)    # Rotate the frame if needed (currently set to 0 degrees)
        frame_HD = frame_HD[480:2590, 50:4000]
        output = frame_HD.copy()
        gray_frame_HD = cv2.cvtColor(frame_HD, cv2.COLOR_BGR2GRAY)    
        blurred_frame_HD = cv2.GaussianBlur(gray_frame_HD, (5, 5), 0)

        # Start dimension calculation
        # Threshold the image
        _, thresholded_obj_detection = cv2.threshold(blurred_frame_HD, 200, 255, cv2.THRESH_BINARY)

        # Apply morphological operations
        closed = cv2.morphologyEx(thresholded_obj_detection, cv2.MORPH_CLOSE, kernel)

        # Inpaint the image to remove glares
        frame_no_glare = cv2.inpaint(frame_HD, closed, 3, cv2.INPAINT_TELEA)

        # Color detection
        # Convert the frame to HSV color space for color-based object detection
        frame_hsv = cv2.cvtColor(frame_no_glare, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(frame_hsv, blue_Lower, blue_Upper)  # Create a mask for blue color
        blue_mask = cv2.bitwise_not(blue_mask)  # Invert the mask

        nonZero_blue = cv2.countNonZero(blue_mask)
        logger.debug(f'1st check: if an object appears on the table (nonZero_blue): {nonZero_blue}')

        # 4th check: if an object appears on the table by checking non-zero pixels in the mask
        if nonZero_blue > 500000:
            logger.debug(f'Object detected with nonZero_blue: {nonZero_blue}')

            # Find contours on the thresholded image
            contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            # Draw all detected contours for visualization
            contour_image = output.copy()
            cv2.drawContours(contour_image, contours, -1, (255, 0, 0), 1)  # Draw contours in blue
            cv2.imwrite(os.path.join(script_dir, 'all_contours.jpg'), contour_image)

            # Initialize flag to track if all contours are skipped
            all_contours_skipped = True
            
            # Loop over the contours to process each detected object
            for cnt in contours:
                area = cv2.contourArea(cnt)
            
                # Skip small contours that likely represent noise
                if area < frame_area * 0.016:
                    logger.debug(f'Contour area: {area} < {frame_area * 0.02}, skipping')
                    continue  

                # Set flag to False since we have at least one valid contour
                all_contours_skipped = False

                # Calculate contour centroid
                M = cv2.moments(cnt)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)

                # Skip contours with too many vertices (likely noise)
                if len(approx) > 8:
                    logger.debug(f'Contour has {len(approx)} vertices, skipping')
                    continue

                # Draw the contour and calculate object dimensions
                cv2.drawContours(output, [approx], 0, (0, 255, 0), 2)
                measurements = cal_dim(contours, hierarchy, pixel_mm_ratio_width, pixel_mm_ratio_height)
                object_width = measurements["width_mm"]
                object_height = measurements["height_mm"]
                object_area = measurements["area_mm2"]

                # Annotate the object area on the image
                cv2.putText(output, "Area: {} mm2".format(round(object_area, 1)), (int(cX - 100), int(cY - 50)),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                
                if len(approx) == 4:
                    logger.debug(f'Contour has 4 vertices, calculating width and height')
                    # Annotate the width and height on the image
                    cv2.putText(output, "Width: {} mm".format(round(object_width, 1)), (int(cX - 100), int(cY - 20)),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                    cv2.putText(output, "Height: {} mm".format(round(object_height, 1)), (int(cX - 100), int(cY + 10)),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                
                # Print the object dimensions to the terminal
                logger.debug("Width: {} mm".format(round(object_width, 1)))
                logger.debug("Height: {} mm".format(round(object_height, 1)))

                (x, y, w, h) = cv2.boundingRect(cnt)

                # Save the processed image with a unique filename
                running_number = get_next_running_number()
                im_fname = os.path.join(img_output_path, f'mtal{running_number}.png')
                cv2.imwrite(im_fname, output)

                # Turn on tower light
                # thread_tl_g = threading.Thread(target=tl.turn_on_pin_a)
                # thread_tl_g.start()

                # Prepare the image file for uploading
                files = prepare_image_file(im_fname)

                # Prepare the request body with object details
                body = prepare_request_body(running_number, object_area, object_width, object_height)
                logger.debug(f'REQUEST: {body}')
                
                # if send_data_to_server(url, files, body, headers, logger):
                if send_data_to_server(url, files, body, headers):
                    status_active = False  # Reset the status flag
            
            # if all_contours_skipped:
            #     thread_tl_b = threading.Thread(target=tl.turn_on_pin_b)
            #     thread_tl_b.start()

        else:
            logger.debug('No object detected, resetting variables')

    # if SHOW_WINDOWS:
    #     cv2.imshow("Raw Frame", frame)

    if SHOW_WINDOWS and 'output' in locals():
        display = cv2.resize(output, (1280, 720))
        cv2.imshow("Processed Output", display)

    if SHOW_WINDOWS:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if SHOW_WINDOWS:
    cap.release()
    cv2.destroyAllWindows()
