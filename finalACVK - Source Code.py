import cv2
import matplotlib.pyplot as plt 
import numpy as np
import argparse

# Menambahkan file cascade untuk deteksi/pengenalan wajah
faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

# Inisialisasi Parser
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

# Inisialisasi Histogram
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


# Inisialisasi Deteksi Tepi
# Ukuran Blur Kernel 
kernelSize=21   

# Parameter Deteksi Tepi
parameter1=20
parameter2=60
intApertureSize=1

# Inisialisasi Webcam
cam = cv2.VideoCapture(0)
cam.set(3,640) # set Lebar
cam.set(4,480) # set Tinggi

while True:
    # Menerima video dari webcam
    ret, img = cam.read()
	
    if not ret:
           break
    k = cv2.waitKey(1)

    # Inisialisasi Deteksi Wajah
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        
        scaleFactor=1.2,
        minNeighbors=5
        ,     
        minSize=(20, 20)
    )

    # Memberi kotak pada Wajah
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

    # Membuat Histogram
    numPixels = np.prod(img.shape[:2])
    if color == 'rgb':
            cv2.imshow('Final Project ACVK - RGB', img)
            (b, g, r) = cv2.split(roi_color )
            histogramR = cv2.calcHist([r], [0], None, [bins], [0, 255]) / numPixels
            histogramG = cv2.calcHist([g], [0], None, [bins], [0, 255]) / numPixels
            histogramB = cv2.calcHist([b], [0], None, [bins], [0, 255]) / numPixels
            lineR.set_ydata(histogramR)
            lineG.set_ydata(histogramG)
            lineB.set_ydata(histogramB)
    else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imshow('Final Project ACVK - Grayscale', gray)
            histogram = cv2.calcHist([roi_gray], [0], None, [bins], [0, 255]) / numPixels
            lineGray.set_ydata(histogram)
    fig.canvas.draw()

    # Membuat Deteksi Tepi
    img = cv2.Canny(img,parameter1,parameter2,intApertureSize)  # Canny edge detection
    cv2.imshow('Deteksi Tepi', img)

    # Tombol Keluar    
    if k & 0xFF == ord('q'):
          break

cap.release()
cv2.destroyAllWindows()