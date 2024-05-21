import numpy as np
import cv2

class ImageProcessor(object):
    def avg_std_color_channels(self,img):
        # Compute the average and standard deviation of each color channel using OpenCV
        avg = np.mean(img, axis=(0, 1))
        std = np.std(img, axis=(0, 1))

        return avg, std

    def adjust_image(self,image, image_avg, image_std):

        # Create arrays for the original mean and standard deviation
        ORIGINAL_AVG = np.array([73.667465,  -2.1255977 , 4.8318872], dtype=image.dtype)
        ORIGINAL_STD = np.array([8.862262 ,   7.953179  , 14.210143], dtype=image.dtype)

        # Compute the adjusted image using OpenCV operations
        t = (image - image_avg) * (ORIGINAL_STD / image_std) + ORIGINAL_AVG
        t = np.clip(t, 0, 255)

        return t

    def color_normalization(self,img):
        # Convert PIL image to NumPy array
        img = np.array(img)
        # Convert image to float32 with range [0, 1]
        img = img.astype(np.float32) / 255.0
        # Convert RGB image to LAB color space using OpenCV
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        # Get measurements
        image_avg, image_std = self.avg_std_color_channels(lab)

        # Adjust image
        processed_lab = self.adjust_image(lab, image_avg, image_std)

        # Convert back to RGB using OpenCV
        processed_bgr = cv2.cvtColor(processed_lab, cv2.COLOR_LAB2BGR)

        return processed_bgr*255

    def white_balance(self, image):
        # Calcular a média dos valores de cada canal de cor
        b, g, r = cv2.split(image)
        b_avg = b.mean()
        g_avg = g.mean()
        r_avg = r.mean()

        # Calcular os fatores de escala para cada canal de cor
        b_scale = r_avg / b_avg
        g_scale = r_avg / g_avg

        # Aplicar os fatores de escala para balancear o branco
        balanced_b = cv2.convertScaleAbs(b, alpha=b_scale, beta=0)
        balanced_g = cv2.convertScaleAbs(g, alpha=g_scale, beta=0)
        balanced_r = r

        # Mesclar novamente os canais de cor balanceados
        balanced_image = cv2.merge((balanced_b, balanced_g, balanced_r))

        return balanced_image

    def normalize_brightness_channel(self, channel, target_brightness=230):
        mean_brightness = np.mean(channel)
        adjustment_ratio = target_brightness / mean_brightness
        channel = np.clip(channel * adjustment_ratio, 0, 255).astype(np.uint8)
        return channel

    def normalize_brightness_HSV(self, image, target_brightness=230):
        image = np.clip(image , 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        image[:, :, 2] = self.normalize_brightness_channel(hsv[:, :, 2], target_brightness=target_brightness)
        return image

    def clahe(self, image):
        lab= cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)

        # Applying CLAHE to L-channel
        # feel free to try different values for the limit and grid size:
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(5,5))
        cl = clahe.apply(l_channel)

        # merge the CLAHE enhanced L-channel with the a and b channel
        limg = cv2.merge((cl,a,b))

        # Converting image from LAB Color model to BGR color spcae
        enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        return enhanced_img

    def sharpen(self, image):
        blurred_img = cv2.GaussianBlur(image, (0, 0), 3)
        sharpened_image = cv2.addWeighted(image, 1.5, blurred_img, -0.5, 0)

        return sharpened_image

    def mean_background(self, image):
        background_min = np.array([0, 0, 230],np.uint8)
        background_max = np.array([180, 50, 255],np.uint8)
        box = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        background_value = np.sum(cv2.inRange(box, background_min, background_max))

        return background_value

    def normalize_background(self, image):
        image = np.array(image)
        image = image[:, :, ::-1]
        background_value = self.mean_background(image)
        #print('back', background_value)
        if background_value > 100000000:
            back_normalized = self.normalize_brightness_HSV(image, target_brightness=230)
        else:
            back_normalized = self.white_balance(image)
        return back_normalized

    def filter_image(self, image):
        img = self.normalize_background(image)
        enhanced_img = self.clahe(img)
        sharpened_image = self.sharpen(enhanced_img)
        norm = self.color_normalization(sharpened_image).astype(np.uint8)
        return norm