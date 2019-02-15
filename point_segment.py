import numpy as np
import cv2
import matplotlib.pyplot as plt

KERNEL = np.array([[-1,-1,-1,-1,-1],
                   [-1,-1,-1,-1,-1],
                   [-1,-1,24,-1,-1],
                   [-1,-1,-1,-1,-1],
                   [-1,-1,-1,-1,-1]])

COLOR = (0, 255, 255)
RADIUS = 10

#Read the image using opencv
def get_image(path):
    return cv2.imread(path)

#Read the image in gray scale using opencv
def get_image_gray(path):
    return cv2.imread(path,0)

#Show the resulting image
def show_image(name, image):
    cv2.imshow(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Save the resulting image
def save_image(name, image):
    cv2.imwrite(name,image) 

#Perform normalization on the image
def norm(image):
    return (image/255).astype(np.uint8)

#Copy the image
def copy(image):
    return image.copy()

#Create a Matrix with all elements 0
def create_zero_matrix(r, c):
    return np.zeros((r,c), dtype='float32')

#Add Zero padding around the edge of the image
def add_padding(image,padding):
    return np.pad(image, (padding,padding), 'edge')

#Perform convolution
def convolution(image,kernel):
    #Get image height and width
    h, w = image.shape[:2]
    
    #Create 3 empty matrix to store the result
    conv_op = create_zero_matrix(h, w)
    
    #Padding width
    pad = kernel.shape[0]//2

    #Kernel Size
    ks = kernel.shape[0]

    #Adding zero padding to the image
    padded_img = add_padding(image, pad)
    
    for i in range(pad, h + pad):
        for j in range(pad, w + pad):
            kx = 0
            #The submatrix from the image on which convolution is done
            #Dimension is based on the dimension of the kernel being applied
            submatrix = padded_img[i-pad:i-pad+ks,j-pad:j-pad+ks]
            
            for x in range(ks):
                for y in range(ks):
                    kx = kx + (kernel[x][y] * submatrix[x][y])
            #Saving the convolved output to a new matrix
            conv_op[i - pad, j - pad] = kx
    return conv_op

def apply_threshold(org, res, T):
    #List to store detected points
    detectedPoints = list()
    #Image Dimensions
    h, w = org.shape[:2]
    for i in range(h):
        for j in range(w):
            if abs(org[i][j])>=T:
                res[i][j] = 255
                detectedPoints.append(tuple([j,i]))
            else:
                res[i][j] = 0
    return res, detectedPoints

#Plot the histogram for the given image
def plot_histogram(image):
    h, w = image.shape[:2]
    H = np.zeros(256)

    for i in range(h):
        for j in range(w):
            H[image[i,j]] += 1
    return H

#Check if the given value is the peak
#by comparing it with its neighbors
def check_peak(H, value, index, r, LT):
    rmax = len(H)
    r1,r2 = 0 if index-r<0 else index-r, rmax if index+r>rmax else index+r
    for i in range(r1,r2):
        if H[i]>value or value<LT:
            return False
    return True

#Find all the possible peaks in the given histogram
def find_peaks(H):
    peaks = list()
    for i in range(len(H)):
        if check_peak(H, H[i], i, 7, 215):
            peaks.append(i)
    return peaks

#Draw a rectangle on the given image using the given points
def draw_rectangle(image, points):
    for p in points:
        cv2.rectangle(image, p[0], p[1], COLOR, 2)
    return image

def main():
    print('__Task 2 (a)__\n')
    print('__Reading the given image : point.jpg__\n')
    img = get_image_gray('point.jpg')
    
    print('__Performing Convolution to find R__\n')
    #Finding R
    R = convolution(img,KERNEL)
    save_image('R.png', R)
    #Finding |R|
    abs_R = np.absolute(R)
    #Finding the max value in |R|
    maximum = np.amax(abs_R)
    #Finding the threshold value
    T = maximum * (.87)
    #Apply the threshold T on the convolved image R
    result, detectedPoints = apply_threshold(R, copy(R), T)
    #Save the unlabled result
    save_image('res_point_unlabled.jpg', result)
    
    print('__No of points detected__  : ', len(detectedPoints))
    
    result = cv2.cvtColor(result,cv2.COLOR_GRAY2RGB)

    #Iterate through all the detected points
    for p in detectedPoints:
        #Draw a circle around the detected point
        cv2.circle(result, p, RADIUS, COLOR, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        x, y = p[0], p[1]
        #Write the coordinates of the detected point as text
        cv2.putText(result,"("+str(x)+","+str(y)+")",(x-10, y+40), font, 0.75, COLOR, 2, cv2.LINE_AA)
    
    #Save the labled result
    save_image('res_point.jpg', result)

    print('\n__Task 2 (b)__\n')

    print('__Reading the given image : segment.jpg__\n')
    simg = get_image_gray('segment.jpg')

    print('__Computing the Histogram of the given image__\n')
    H = plot_histogram(simg)
    #Ignore the background of the image to have a better view of the
    #foreground distribution in the image
    H[0] = 0
    plt.plot(H, color='m')
    plt.title('segment.jpg Histogram')
    plt.grid(True)
    plt.savefig('segment_histogram.png')
    
    peaks = find_peaks(H)
    
    maxForegroundPeak = peaks[-1:][0]
    threshold = maxForegroundPeak * .97
    print('__Optimal threshold from histogram__ : ', threshold, '\n')
    result, dp = apply_threshold(simg, copy(simg), threshold)
    points = [[(162,125),(201,165)],[(253,77),(300,204)],[(329,24),(365,287)],[(387,41),(423,251)]]
    result = cv2.cvtColor(result,cv2.COLOR_GRAY2RGB)
    result = draw_rectangle(result, points)
    save_image('res_segment.jpg', result)



if __name__ == '__main__':
    main()
