import cv2
import numpy as np
from matplotlib import pyplot as plt

# # 生成带有高斯噪声的示例图像
# # np.random.seed(0)
# # image = np.zeros((512, 512), dtype=np.float32)
# # cv2.randn(image, 0, 50)  # 添加高斯噪声，标准差为50
# # 读取图片
# image_path = '/home/majc/TTA/plug-and-play/data/ILSVRC2012_val_00022597.JPEG'
# image = cv2.imread(image_path, cv2.IMREAD_COLOR)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将BGR格式转换为RGB格式

# # 将图像转换为浮点数格式进行处理
# image_float = image.astype(np.float32) / 255.0

# # 傅里叶变换
# f_image = np.fft.fft2(image_float, axes=(0, 1))
# fshift = np.fft.fftshift(f_image, axes=(0, 1))

# # 构建频域滤波器（例如高斯低通滤波器）
# rows, cols, channels = image.shape
# print(image.shape)
# crow, ccol = rows // 2 , cols // 2  # 中心点位置
# d = 50  # 滤波器半径
# mask = np.zeros((rows, cols, channels), np.uint8)
# for channel in range(channels):
#     crow, ccol = rows // 2 , cols // 2  # 中心点位置
#     mask[crow-d:crow+d, ccol-d:ccol+d, channel] = 1

# # 应用滤波器
# fshift_filtered = fshift * mask

# # 傅里叶逆变换
# f_ishift = np.fft.ifftshift(fshift_filtered, axes=(0, 1))
# image_filtered_float = np.fft.ifft2(f_ishift, axes=(0, 1))
# image_filtered = np.abs(image_filtered_float)
# image_filtered = np.clip(image_filtered, 0, 1)  # 限制在0到1之间

# # 将图像恢复为0-255整数类型，并转换为BGR格式
# image_filtered = (image_filtered * 255).astype(np.uint8)
# image_filtered = cv2.cvtColor(image_filtered, cv2.COLOR_RGB2BGR)

# # 保存滤波后的图像
# filtered_image_path = '/home/majc/TTA/plug-and-play/data/filtered_image.jpg'
# cv2.imwrite(filtered_image_path, image_filtered)


def apply_filter_in_fourier_space(image):
    # Convert image to float32
    image = np.float32(image)
    
    # Initialize filtered image
    filtered_image = np.zeros_like(image)

    # Perform Fourier Transform for each channel
    for i in range(3):  # Assuming image has 3 channels (BGR)
        dft_image = cv2.dft(image[:, :, i], flags=cv2.DFT_COMPLEX_OUTPUT)

        # Shift the zero-frequency component to the center of the spectrum
        dft_image_shifted = np.fft.fftshift(dft_image)

        # Create a mask with high values for noise frequencies
        rows, cols = image.shape[:2]
        crow, ccol = rows // 2, cols // 2

        # Design a filter based on the noisy image's Fourier transform
        mask = np.zeros((rows, cols, 2), np.uint8)
        r = 30  # Radius of the filter
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0])**2 + (y - center[1])**2 <= r*r
        mask[mask_area] = 1

        # Apply the mask to the image's Fourier transform
        fshift = dft_image_shifted * mask

        # Shift back and inverse DFT
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        # Normalize the result
        cv2.normalize(img_back, img_back, 0, 1, cv2.NORM_MINMAX)

        # Store the filtered channel
        filtered_image[:, :, i] = img_back

    return filtered_image

# def apply_dct_filter(image, threshold=10):
#     # Convert image to float32 and normalize
#     image_float = image.astype(np.float32) / 255.0
    
#     # Apply DCT to each channel
#     dct_image = cv2.dct(image_float)

#     # Get dimensions of the image
#     rows, cols, _ = image.shape
    
#     # Create a mask to select bottom right quarter
#     mask = np.zeros_like(dct_image, dtype=np.float32)
#     mask[rows//2:, cols//2:, :] = 1
    
#     # Apply mask to DCT coefficients
#     masked_dct = dct_image * mask
    
#     # Thresholding in masked area
#     masked_dct[np.abs(masked_dct) < threshold] = 0
    
#     # Restore original DCT coefficients in the masked area
#     restored_dct = dct_image * (1.0 - mask) + masked_dct
    
#     # Apply inverse DCT
#     filtered_image_float = cv2.idct(restored_dct)
    
#     # Convert back to uint8
#     filtered_image = (filtered_image_float * 255).astype(np.uint8)
    
#     return filtered_image
def apply_dct_filter(image, threshold=10):
    # Convert image to float32 and normalize
    image = np.float32(image) / 255.0
    
    # Split channels for DCT
    channels = cv2.split(image)
    dct_channels = []
    
    # Apply DCT to each channel
    for channel in channels:
        # Apply DCT
        dct = cv2.dct(np.float32(channel))
        dct[np.abs(dct) < threshold] = 0
s
        idct = cv2.idct(dct)
        
        dct_channels.append(idct)
    
    # Merge channels back together
    filtered_image = cv2.merge(dct_channels)
    
    # Convert back to uint8
    filtered_image = filtered_image * 255
    
    return filtered_image


# Load image
image_path = '/home/majc/TTA/plug-and-play/data/ILSVRC2012_val_00022597.JPEG'  
image = cv2.imread(image_path)
# 应用高斯滤波
# filtered_image = cv2.GaussianBlur(image, (3, 3), 0)  # (5, 5) 是高斯核的大小，可以根据需要调整

# filtered_image = apply_filter_in_fourier_space(image)
filtered_image = apply_dct_filter(image,0.5)

# Save results
cv2.imwrite('/home/majc/TTA/plug-and-play/data/original_image.png', image)
cv2.imwrite('/home/majc/TTA/plug-and-play/data/filtered_image.png', filtered_image)


