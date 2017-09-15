import cv2
import matplotlib.pyplot as plt

def preprocess_image(img, mode="train", blur_px=3):
    """

    :param img: image to be processed
    :param mode: can be either "train" or "drive"
    :return: the processed image
    """

    #if True:
    #    plt.imshow(lum_img)

    # apply subtle blur
    new_img = cv2.GaussianBlur(img, (blur_px, blur_px), 0)
    # scale to 66x200x3 (same as nVidia)
    #new_img = cv2.resize(new_img,(200, 66), interpolation=cv2.INTER_AREA)

    # convert to YUV color space (as nVidia paper suggests)
    if mode == "train":
        new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2YUV)
    elif mode == "drive":
        new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2YUV)

    #if True:
    #    plt.imshow(new_img)

    return new_img
