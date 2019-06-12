#HOW TO RUN
#python [file_name] || if you want to run it with RGB color then : python [file_name] -c rgb

import cv2
import matplotlib.pyplot as plt 
import numpy as np
import argparse
from skimage.exposure import rescale_intensity

# parser inisialisasi
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--color', type=str, default='gray',
    help='Color space: "gray" (default), "rgb"')
parser.add_argument('-b', '--bins', type=int, default=16,
    help='Number of bins per channel (default 16)')
parser.add_argument('-w', '--width', type=int, default=0,
    help='Resize video to specified width in pixels (maintains aspect)')
args = vars(parser.parse_args())

color = args['color']
bins = args['bins']
resizeWidth = args['width']

#############################################################################CONVOLUTION
def convolve(image, kernel):
    # grab the spatial dimensions of the image, along with
    # the spatial dimensions of the kernel
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]

    # allocate memory for the output image, taking care to
    # "pad" the borders of the input image so the spatial
    # # size (i.e., width and height) are not reduced
    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
        cv2.BORDER_REPLICATE)

    output = np.zeros((iH, iW), dtype="float32")
    # loop over the input image, "sliding" the kernel across
    # each (x, y)-coordinate from left-to-right and top to
    # bottom
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            # extract the ROI of the image by extracting the
            # *center* region of the current (x, y)-coordinates
            # dimensions
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
            
            # perform the actual convolution by taking the
            # element-wise multiplicate between the ROI and
            # the kernel, then summing the matrix
            k = (roi * kernel).sum()
            
            # store the convolved value in the output (x,y)-
            # coordinate of the output image
            output[y - pad, x - pad] = k
    # rescale the output imag   e to be in the range [0, 255]
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")
            
    # return the output image
    return output

# construct average blurring kernels used to smooth an image
smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))
                
# construct a sharpening filter
sharpen = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]), dtype="int")
# construct the Laplacian kernel used to detect edge-like
# regions of an image
laplacian = np.array((
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]), dtype="int")
                
# construct the Sobel x-axis kernel
sobelX = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]), dtype="int")
                
# construct the Sobel y-axis kernel
sobelY = np.array((
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]), dtype="int")
                
# construct the kernel bank, a list of kernels we're going
# to apply using both our custom `convole` function and
# OpenCV's `filter2D` function
kernelBank = (
    ("small_blur", smallBlur),
    ("large_blur", largeBlur),
    ("sharpen", sharpen),
    ("laplacian", laplacian),
    ("sobel_x", sobelX),
    ("sobel_y", sobelY)
)

cam = cv2.VideoCapture(0)

################################################################################################HISTOGRAM
# inisialisasi plot
fig, ax = plt.subplots()
if color == 'rgb':
    ax.set_title('Histogram (RGB)')
else:
    ax.set_title('Histogram (grayscale)')
ax.set_xlabel('Bin')
ax.set_ylabel('Frequency')

#inisialisasi plot
lw = 3
alpha = 0.5
if color == 'rgb':
    lineR, = ax.plot(np.arange(bins), np.zeros((bins,)), c='r', lw=lw, alpha=alpha, label='Red')
    lineG, = ax.plot(np.arange(bins), np.zeros((bins,)), c='g', lw=lw, alpha=alpha, label='Green')
    lineB, = ax.plot(np.arange(bins), np.zeros((bins,)), c='b', lw=lw, alpha=alpha, label='Blue')
else:
    lineGray, = ax.plot(np.arange(bins), np.zeros((bins,1)), c='k', lw=lw, label='intensity')
ax.set_xlim(0, bins-1)
ax.set_ylim(0, 1)
ax.legend()
plt.ion()
plt.show()

#initialization
img_counter = 0

while True:
    (ret, frame) = cam.read()
    if not ret:
        break
    k = cv2.waitKey(1)

    numPixels = np.prod(frame.shape[:2])

    # Resize frame to width, if specified.
    if resizeWidth > 0:
        (height, width) = frame.shape[:2]
        resizeHeight = int(float(resizeWidth / width) * height)
        frame = cv2.resize(frame, (resizeWidth, resizeHeight),
            interpolation=cv2.INTER_AREA)
    numPixels = np.prod(frame.shape[:2])
    if color == 'rgb':
        cv2.imshow('Final Project ACVK - RGB', frame)
        (b, g, r) = cv2.split(frame)
        histogramR = cv2.calcHist([r], [0], None, [bins], [0, 255]) / numPixels
        histogramG = cv2.calcHist([g], [0], None, [bins], [0, 255]) / numPixels
        histogramB = cv2.calcHist([b], [0], None, [bins], [0, 255]) / numPixels
        lineR.set_ydata(histogramR)
        lineG.set_ydata(histogramG)
        lineB.set_ydata(histogramB)
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Final Project ACVK - Grayscale', gray)
        histogram = cv2.calcHist([gray], [0], None, [bins], [0, 255]) / numPixels
        lineGray.set_ydata(histogram)
    fig.canvas.draw()

    if k & 0xFF == ord('q'):
            # ESC pressed
            print("q ditekan, menutup program...")
            break

    elif k & 0xFF == ord('i'):
            # SPACE pressed
            img_name = "gambar_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1
    
    elif k & 0xFF == ord('h'):
            # inisialisasi plot
            fig, ax = plt.subplots()
            if color == 'rgb':
                ax.set_title('Histogram (RGB)')
            else:
                ax.set_title('Histogram (grayscale)')
            ax.set_xlabel('Bin')
            ax.set_ylabel('Frequency')

            #inisialisasi plot
            lw = 3
            alpha = 0.5
            if color == 'rgb':
                lineR, = ax.plot(np.arange(bins), np.zeros((bins,)), c='r', lw=lw, alpha=alpha, label='Red')
                lineG, = ax.plot(np.arange(bins), np.zeros((bins,)), c='g', lw=lw, alpha=alpha, label='Green')
                lineB, = ax.plot(np.arange(bins), np.zeros((bins,)), c='b', lw=lw, alpha=alpha, label='Blue')
            else:
                lineGray, = ax.plot(np.arange(bins), np.zeros((bins,1)), c='k', lw=lw, label='intensity')
            ax.set_xlim(0, bins-1)
            ax.set_ylim(0, 1)
            ax.legend()
            plt.ion()
            plt.show()
            numPixels = np.prod(frame.shape[:2])

            # Resize frame to width, if specified.
            if resizeWidth > 0:
                (height, width) = frame.shape[:2]
                resizeHeight = int(float(resizeWidth / width) * height)
                frame = cv2.resize(frame, (resizeWidth, resizeHeight),
                    interpolation=cv2.INTER_AREA)
            numPixels = np.prod(frame.shape[:2])
            if color == 'rgb':
                cv2.imshow('Final Project ACVK - RGB', frame)
                (b, g, r) = cv2.split(frame)
                histogramR = cv2.calcHist([r], [0], None, [bins], [0, 255]) / numPixels
                histogramG = cv2.calcHist([g], [0], None, [bins], [0, 255]) / numPixels
                histogramB = cv2.calcHist([b], [0], None, [bins], [0, 255]) / numPixels
                lineR.set_ydata(histogramR)
                lineG.set_ydata(histogramG)
                lineB.set_ydata(histogramB)
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imshow('Final Project ACVK - Grayscale', gray)
                histogram = cv2.calcHist([gray], [0], None, [bins], [0, 255]) / numPixels
                lineGray.set_ydata(histogram)
            fig.canvas.draw()
            


    elif k & 0xFF == ord('o'):
            edgeImage = input('input nama file :')
            img = cv2.imread(edgeImage,0)
            edges = cv2.Canny(img,100,200)
            plt.show()
            
            plt.subplot(121),plt.imshow(img,cmap = 'gray')
            plt.title('Original Image'), plt.xticks([]), plt.yticks([])
            plt.subplot(122),plt.imshow(edges,cmap = 'gray')
            plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    elif k & 0xFF == ord('m'):
            morphImage = input('input nama file :')
            img = cv2.imread(morphImage,0)

            plt.ion()
            plt.show()

            # get binary image and apply Gaussian blur
            imageGray = img
            imagePreprocessed = cv2.GaussianBlur(imageGray, (5, 5), 0)
            plt.subplot(321), plt.imshow(cv2.cvtColor(imagePreprocessed, cv2.COLOR_GRAY2RGB))
            plt.title('Preprocessed Image'), plt.xticks([]), plt.yticks([])

            _, imageBinary = cv2.threshold(imagePreprocessed, 130, 255, cv2.THRESH_BINARY)

            # invert gambar untuk mendapatkan objek
            imageBinary = cv2.bitwise_not(imageBinary)

            # Morfologi kernel
            morphKernel = np.ones((15,15),np.uint8)

            # Erosi
            imageMorph = cv2.erode(imageBinary, morphKernel, iterations=2)
            plt.subplot(322),plt.imshow(cv2.cvtColor(imageMorph, cv2.COLOR_GRAY2RGB))
            plt.title('Erode Image'), plt.xticks([]), plt.yticks([])

            # Dilasi
            imageMorph = cv2.dilate(imageBinary, morphKernel, iterations=2)
            plt.subplot(323),plt.imshow(cv2.cvtColor(imageMorph, cv2.COLOR_GRAY2RGB))
            plt.title('Dilate Image'), plt.xticks([]), plt.yticks([])

            # Final Image
            imageMorph = cv2.morphologyEx(imageBinary, cv2.MORPH_CLOSE, morphKernel)

            plt.subplot(324),plt.imshow(cv2.cvtColor(imageMorph, cv2.COLOR_GRAY2RGB))
            plt.title('Final Image'), plt.xticks([]), plt.yticks([])

    elif k & 0xFF == ord('c'):
            image = cv2.imread(convolutionImage)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  

            # loop over the kernels
            for (kernelName, kernel) in kernelBank:
                # apply the kernel to the grayscale image using both
                # our custom `convole` function and OpenCV's `filter2D`
                # function
                print("[INFO] applying {} kernel".format(kernelName))
                convoleOutput = convolve(gray, kernel)
                opencvOutput = cv2.filter2D(gray, -1, kernel)
            
                # show the output images
                cv2.imshow("original", gray)
                cv2.imshow("{} - convole".format(kernelName), convoleOutput)
                cv2.imshow("{} - opencv".format(kernelName), opencvOutput)

cam.release()

cv2.destroyAllWindows()