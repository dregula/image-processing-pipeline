import os
from ast import literal_eval as le

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2

os.chdir('D:/Whole_Slide/SlideTracking/HE99E1_LZW_ObjectiveConvertor')

split_tiffs = [
    # "HE_99E1_bigtiff_RGB_265x180.tif",
    # "HE_99E1_bigtiff_RGB_530x360.tif",
    # "HE_99E1_bigtiff_RGB_715x710_cropped.tif",
    # "HE_99E1_bigtiff_RGB_1424x1440_cropped.tif",
    # "HE_99E1_bigtiff_RGB_1848x2880_cropped.tif",
    "HE_99E1_bigtiff_RGB_5728x57280_cropped.tif",
    "HE_99E1_bigtiff_RGB_22848x22912_cropped.tif",
    "HE_99E1_bigtiff_RGB_45504x45992_cropped.tif",
    "HE_99E1_bigtiff_RGB_91264x91648_cropped.tif"   # this "40x" page doesn't open with anything (and may be a pseudo-=image anyway, as the iSyntax claims a 20x base magnification)
]

for idx, split_tiff in enumerate(split_tiffs):
    img = cv2.imread(split_tiff)
    print(img.shape)
    # cv2.imshow(str(idx), img)
    # cv2.waitKey()
