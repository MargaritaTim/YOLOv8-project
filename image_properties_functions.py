''' image properties functions '''

# imports
import cv2
from cv2 import IMREAD_COLOR, IMREAD_UNCHANGED

from PIL import Image, ImageStat

# useful packeges
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# statistic packeges
from scipy.ndimage import variance
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import laplace
from skimage.transform import resize


# %matplotlib inline

""" aspect ratio (width-height) """
def return_aspect_ratio(w,h):
  return float(w) / h

"""brightness"""

# Convert the image to grayscale
def convert_image_to_grayscale(image):
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  return gray_image

# Calculate the mean brightness value
def get_image_brightness(image):
  brightness = int(round(cv2.mean(image)[0]))
  return brightness
#source: https://stackoverflow.com/questions/3490727/what-are-some-methods-to-analyze-image-brightness-using-python


"""perceived brightness"""



"""contrast

https://stackoverflow.com/questions/58821130/how-to-calculate-the-contrast-of-an-image
"""


def contrast(image_path):
  # load image as YUV (or YCbCR) and select Y (intensity)
  # or convert to grayscale, which should be the same.
  # Alternately, use L (luminance) from LAB.
  img = cv2.imread(image_path)
  Y = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:, :, 0]

  # compute min and max of Y
  min = np.min(Y)
  max = np.max(Y)

  # compute contrast
  contrast = (max - min) / (max + min)
  return contrast


"""blur
https://www.kaggle.com/code/eladhaziza/perform-blur-detection-with-opencv
"""



def variance_of_laplacian(img2):
  # compute the Laplacian of the image and then return the focus
  # measure, which is simply the variance of the Laplacian
  gray = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
  return cv2.Laplacian(gray, cv2.CV_64F).var()


def BGR2RGB(BGR_img):
  # turning BGR pixel color to RGB
  rgb_image = cv2.cvtColor(BGR_img, cv2.COLOR_BGR2RGB)
  return rgb_image


def blurrinesDetection(directories, threshold):
  columns = 3
  rows = len(directories) // 2
  fig = plt.figure(figsize=(5 * columns, 4 * rows))
  for i, directory in enumerate(directories):
    fig.add_subplot(rows, columns, i + 1)
    img = cv2.imread(directory)
    text = "Not Blurry"
    # if the focus measure is less than the supplied threshold,
    # then the image should be considered "blurry
    fm = variance_of_laplacian(img)
    if fm < threshold:
      text = "Blurry"
    rgb_img = BGR2RGB(img)
    cv2.putText(rgb_img, "{}: {:.2f}".format(text, fm), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    plt.imshow(rgb_img)
  plt.show()

  def num_of_edges_in_photo(image_path, lower_threshold, higher_threshold):
    # Load the image
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection algorithm
    edges = cv2.Canny(gray, lower_threshold, higher_threshold)

    # Count the number of edges
    num_edges = cv2.countNonZero(edges)

    # Print the number of edges
    return num_edges

  def ppi_resolution(image_path):
      from PIL import Image
      from fractions import Fraction

      # Load the image
      img = Image.open(image_path)

      # Extract the DPI information from the EXIF data
      dpi_x, dpi_y = img.info.get('dpi', (None, None))

      # Extract the physical size information from the EXIF data
      x_res, y_res = img.info.get('xresolution', None), img.info.get('yresolution', None)
      if x_res and y_res:
        x_res, y_res = Fraction(x_res[0], x_res[1]), Fraction(y_res[0], y_res[1])
        dpi_x, dpi_y = float(x_res), float(y_res)

      # Calculate the physical size of the image in inches
      if dpi_x and dpi_y:
        width_in = img.size[0] / dpi_x

      # Load the image in cv2
      img = cv2.imread(image_path)

      # Get the image size in pixels
      width_px, height_px = img.shape[:2]

      # Calculate the ppi/dpi of the image
      ppi = round(width_px / width_in)

      # return the results
      return ppi