import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# ----- 지침
# 다양한 각도로 체스보드 16장 찍기 (체스보드가 평평하게 놓여 있다는 가정 하에)
# -> 모니터에 띄우두고 카메라로 찍으면 훨씬 편함
# -> 휴대폰으로 찍을 때 광량, 포커싱, 조리개 값 (보통 작게) 고정, ISO 값 높이기 -> 이값들 고정 시키고 촬영해야함
# 영상 해상도 1024 * 768 HD
# 격자의 크기를 자로 정확하게 제기 (mm기준)


def plot_img(rows, cols, index, img, title, axis='on'):
    ax = plt.subplot(rows, cols, index)
    if(len(img.shape)==3):
        ax_img = plt.imshow(img[...,::-1])
    else:
        ax_img = plt.imshow(img, cmap='gray')
    
    plt.axis(axis)
    if(title != None): plt.title(title)
    return  ax_img, ax

def display_untilKey(Pimgs, Titles, file_out = False):
    for img, title in zip(Pimgs, Titles):
        cv.imshow(title, img)
        if file_out == True:
            cv.imwrite(title + '.jpeg', img)
    cv.waitKey(0)
    
def detect_2d_points_from_cbimg(file_name, pattern_size):
    img = cv.imread(file_name)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    found, corners = cv.findChessboardCorners(img_gray, pattern_size)
    
    if found:
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1) # 0.001
        corners = cv.cornerSubPix(img_gray, corners, (5,5), (-1, -1), criteria)
        cv.drawChessboardCorners(img, pattern_size, corners, found)
        
    if not found:
        print('chessboard not found')
        return None
    
    return (corners, img)


if __name__ == '__main__':
    pattern_size = (9,6)
    square_size = 22 # mm 기준 
    
    '''
    pattern_points = []
    idx = 0
    for y in range(pattern_size[1]):
        for x in range(pattern_size[0]):
            pattern_points.append((x,y,0))
            idx += 1
    
    pattern_points = np.array(pattern_points).reshape(-1, 3).astype(np.float32)
    '''
    
    pattern_points = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    
    pattern_points *= square_size
    
    import glob
    image_names = glob.glob('./my_calib_resize_img/*.jpeg')
    print(image_names)
    
    chessboard_imgs = []
    chessboard_imgs = [ detect_2d_points_from_cbimg(file_name, pattern_size) for file_name in image_names ]
    
    # Arrays to store object points and image points from all the images.
    obj_points = [] # 3d point in real world space
    img_points = [] # 2d points in image plane.
    idx = 1
    #plt.figure()
    for x in chessboard_imgs:
        if x is not None:
            img_points.append(x[0])
            obj_points.append(pattern_points)
            plot_img(5, 5, idx, x[1], None, 'off')
            idx += 1
    
    #plt.show()
    
    h, w = cv.imread(image_names[0]).shape[:2]
    print('image size : %d' %w + ', %d' %h + ' %d images' %len(image_names))
    
    rms_err, intrisic_mtx, dist_coefs, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, (w,h), None, None)
    
    print('\nRMS: ', rms_err)
    print('camera intrinsic matrix: \n', intrisic_mtx)
    print('distortion coefficients: ', dist_coefs.ravel())
    
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(intrisic_mtx, dist_coefs, (w,h), 1, (w,h))
#    newcameramtx, roi = cv.getOptimalNewCameraMatrix(intrisic_mtx, dist_coefs, (w,h), 0, (w,h))
    print('camera new intrinsic matrix:\n', newcameramtx)
    
    # store the camera parameters
    # https://docs.opencv.org/master/dd/d74/tutorial_file_input_output_with_xml_yml.html
    fs = cv.FileStorage('./camera_parameters.txt', cv.FileStorage_WRITE)
    fs.write('camera intrinsic matrix', intrisic_mtx)
    fs.write('distortion coefficient', dist_coefs)
    fs.write('camera new intrinsic matrix', newcameramtx)
    fs.release()
    
    img_test = cv.imread(image_names[9])
    # undistort
    img_undist_test = cv.undistort(img_test, intrisic_mtx, dist_coefs, None, newcameramtx)
        
    plt.figure(2)
    plot_img(1, 2, 1, img_test, 'distorted', 'off')
    plot_img(1, 2, 2, img_undist_test, 'undistorted', 'off')
    plt.show()
    
    