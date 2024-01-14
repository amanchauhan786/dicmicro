import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, convolve2d
from skimage import io, color, transform
from scipy.ndimage import gaussian_filter
from scipy.stats import zscore
from scipy import signal
import streamlit as st

# Background
def set_bg_hack_url():
    st.markdown(
        f"""
         <style>
         .stApp {{
             background: url("https://images3.alphacoders.com/108/1082058.jpg");
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True
    )
set_bg_hack_url()

def remove_halo_effect(image, kernel_size=3, contrast_factor=1.5):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the image to reduce the halo
    blurred_image = cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)

    # Calculate the difference between the original and blurred images
    halo_difference = gray_image - blurred_image

    # Adjust contrast by amplifying the halo difference
    adjusted_difference = gray_image + contrast_factor * halo_difference

    # Clip the adjusted values to the valid range [0, 255]
    adjusted_image = np.clip(adjusted_difference, 0, 255).astype(np.uint8)

    # Create a color image by replicating the adjusted channel to three channels
    result_image = cv2.cvtColor(adjusted_image, cv2.COLOR_GRAY2BGR)

    return result_image

def resize_image(image, target_shape):
    return transform.resize(image, target_shape, anti_aliasing=True, preserve_range=True).astype(np.uint8)

def compare_images(image1, image2):
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    similarity_index = signal.correlate2d(gray_image1, gray_image2, mode='same')
    return np.max(similarity_index)

def richardson_lucy(image, psf, iterations=30):
    # Richardson-Lucy deconvolution
    estimate = np.full_like(image, 0.5)

    # Convert image and psf to 2D arrays if they are not already
    image = np.atleast_2d(image)
    psf = np.atleast_2d(psf)

    psf_mirror = psf[::-1, ::-1]

    try:
        for _ in range(iterations):
            for channel in range(image.shape[2]):
                # Convert estimate to 2D array
                estimate_channel = np.atleast_2d(estimate[:, :, channel])

                relative_blur = image[:, :, channel] / convolve2d(estimate_channel, psf, 'same', 'symm')
                estimate_channel *= convolve2d(relative_blur, psf_mirror, 'same', 'symm')

    except ValueError as e:
        st.warning(f"Error during deconvolution: {e}")
        # You can add additional error handling or return a default value if needed
        return np.zeros_like(image)

    return estimate

def main():
    st.title("Image Processing with Streamlit")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = io.imread(uploaded_file)

        result_image = remove_halo_effect(image)

        st.image([image, result_image], caption=['Original Image', 'Halo-Removed Image'], width=300)

        hilbert_transformed = np.imag(hilbert(image))
        hilbert_transformed = (hilbert_transformed - np.min(hilbert_transformed)) / (np.max(hilbert_transformed) - np.min(hilbert_transformed))
        psf = np.ones(hilbert_transformed.shape) / np.prod(hilbert_transformed.shape)

        # Use richardson_lucy from scipy for deconvolution
        # deconvolved = richardson_lucy(hilbert_transformed, psf, iterations=30)
        #
        # st.image([image, hilbert_transformed, deconvolved],
        #          caption=['Original Image', 'Hilbert-Transformed Image', 'Deconvolved Image'],
        #          width=300)
        #
        # st.write(f'Similarity Index (Hilbert-transformed vs Uploaded Image): {compare_images(hilbert_transformed, image)}')

if __name__ == "__main__":
    main()
