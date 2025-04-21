import streamlit as st
import cv2
import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma, denoise_tv_chambolle
from skimage import img_as_float, img_as_ubyte
from skimage.exposure import adjust_gamma, equalize_adapthist, rescale_intensity
from skimage.util import random_noise
import pywt
from PIL import Image
import io
import time
import threading
import queue
import os
from datetime import datetime


def add_noise(image, noise_type="gaussian", amount=0.05):
    noisy_img = random_noise(img_as_float(image), mode=noise_type, var=amount)
    return img_as_ubyte(noisy_img)


def gaussian_denoise(image, ksize=5):
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def median_denoise(image, ksize=5):
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.medianBlur(image, ksize)


def bilateral_denoise(image, d=9, sigmaColor=75, sigmaSpace=75):
    return cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)


def non_local_means_denoise(image, h_param=1.15, patch_size=5, patch_distance=7):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sigma_est = np.mean(estimate_sigma(image_gray))
    
    result = np.zeros_like(image)
    for i in range(3):
        channel = image[:, :, i]
        denoised_channel = denoise_nl_means(
            img_as_float(channel),
            h=h_param * sigma_est,
            fast_mode=True,
            patch_size=patch_size,
            patch_distance=patch_distance
        )
        result[:, :, i] = img_as_ubyte(denoised_channel)
    
    return result


def wavelet_denoising(image, wavelet='db1', threshold=0.04):
    result = np.zeros_like(image)
    
    for i in range(3):
        channel = image[:, :, i]
        
        coeffs = pywt.wavedec2(channel, wavelet, level=2)
        
        coeffs_thresholded = list(coeffs)
        for j in range(1, len(coeffs_thresholded)):
            coeffs_thresholded[j] = tuple(pywt.threshold(c, threshold*np.max(channel), mode='soft') for c in coeffs_thresholded[j])
        
        denoised_channel = pywt.waverec2(coeffs_thresholded, wavelet)
        denoised_channel = denoised_channel[:channel.shape[0], :channel.shape[1]]
        
        result[:, :, i] = np.clip(denoised_channel, 0, 255).astype(np.uint8)
    
    return result


def total_variation_denoising(image, weight=0.1):
    img_float = img_as_float(image)
    denoised = denoise_tv_chambolle(img_float, weight=weight, channel_axis=2)
    return img_as_ubyte(denoised)


def anisotropic_diffusion(image, num_iter=10, delta_t=0.1, kappa=50):
    img = np.float32(image)
    
    for i in range(3):
        channel = img[:, :, i]
        for _ in range(num_iter):
            diff = cv2.Laplacian(channel, cv2.CV_32F)
            channel += delta_t * diff / (1 + np.abs(diff / kappa))
        img[:, :, i] = channel
    
    return np.uint8(np.clip(img, 0, 255))


def super_resolution(image, scale=2):
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def gamma_correction(image, gamma=1.0):
    gamma_corrected = adjust_gamma(image, gamma)
    return img_as_ubyte(gamma_corrected)


def contrast_stretching(image, percentile_low=2, percentile_high=98):
    p_low, p_high = np.percentile(image, (percentile_low, percentile_high))
    stretched = rescale_intensity(image, in_range=(p_low, p_high))
    return img_as_ubyte(stretched)


def color_histogram_equalization(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    v_eq = cv2.equalizeHist(v)
    
    hsv_eq = cv2.merge([h, s, v_eq])
    return cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)


def clahe_enhancement(image, clip_limit=3.0, tile_size=(8, 8)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    cl = clahe.apply(l)
    
    enhanced_lab = cv2.merge([cl, a, b])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def sharpening(image, strength=1.0):
    kernel = np.array([[0, -1, 0], [-1, 4 + strength, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def detail_enhancement(image, sigma_s=10, sigma_r=0.15):
    return cv2.detailEnhance(image, sigma_s=sigma_s, sigma_r=sigma_r)


def process_image(image, denoising_method, enhancement_method, params, result_queue):
    try:
        method_name = denoising_method.lower().replace(" ", "_")
        if method_name == "gaussian_blur":
            denoised = gaussian_denoise(image, params.get("ksize", 5))
        elif method_name == "median_blur":
            denoised = median_denoise(image, params.get("ksize", 5))
        elif method_name == "bilateral_filter":
            denoised = bilateral_denoise(
                image, 
                params.get("d", 9), 
                params.get("sigma_color", 75), 
                params.get("sigma_space", 75)
            )
        elif method_name == "non-local_means":
            denoised = non_local_means_denoise(
                image,
                params.get("h_param", 1.15),
                params.get("patch_size", 5),
                params.get("patch_distance", 7)
            )
        elif method_name == "wavelet_denoising":
            denoised = wavelet_denoising(
                image, 
                params.get("wavelet", "db1"), 
                params.get("threshold", 0.04)
            )
        elif method_name == "total_variation_denoising":
            denoised = total_variation_denoising(image, params.get("weight", 0.1))
        elif method_name == "anisotropic_diffusion":
            denoised = anisotropic_diffusion(
                image,
                params.get("num_iter", 10),
                params.get("delta_t", 0.1),
                params.get("kappa", 50)
            )
        else:
            denoised = image.copy()

        method_name = enhancement_method.lower().replace(" ", "_")
        if method_name == "none":
            enhanced = denoised
        elif method_name == "histogram_equalization":
            enhanced = color_histogram_equalization(denoised)
        elif method_name == "clahe":
            enhanced = clahe_enhancement(
                denoised, 
                params.get("clip_limit", 3.0),
                params.get("tile_size", (8, 8))
            )
        elif method_name == "sharpening":
            enhanced = sharpening(denoised, params.get("strength", 1.0))
        elif method_name == "super-resolution":
            enhanced = super_resolution(denoised, params.get("scale", 2))
        elif method_name == "gamma_correction":
            enhanced = gamma_correction(denoised, params.get("gamma", 1.0))
        elif method_name == "contrast_stretching":
            enhanced = contrast_stretching(
                denoised,
                params.get("percentile_low", 2),
                params.get("percentile_high", 98)
            )
        elif method_name == "detail_enhancement":
            enhanced = detail_enhancement(
                denoised, 
                params.get("sigma_s", 10), 
                params.get("sigma_r", 0.15)
            )
        else:
            enhanced = denoised

        result_queue.put(enhanced)
    except Exception as e:
        result_queue.put(f"Error: {str(e)}")


def get_from_cache(cache_key):
    if 'cache' not in st.session_state:
        st.session_state.cache = {}
    return st.session_state.cache.get(cache_key)


def save_to_cache(cache_key, value):
    if 'cache' not in st.session_state:
        st.session_state.cache = {}
    st.session_state.cache[cache_key] = value


def get_filename_without_extension(file):
    if file is not None:
        filename = file.name
        return os.path.splitext(filename)[0]
    return "image"

st.set_page_config(layout="wide", page_title="Advanced Image Restoration")

st.title("🖼️ Advanced Image Restoration and Denoising App")

if 'processing_history' not in st.session_state:
    st.session_state.processing_history = []
    
if 'current_state_index' not in st.session_state:
    st.session_state.current_state_index = -1


with st.sidebar:
    st.header("Upload Image")
    uploaded_files = st.file_uploader("Upload Image(s)", type=["jpg", "jpeg", "png", "bmp", "tiff"], accept_multiple_files=True)
    
    if uploaded_files:
        if len(uploaded_files) > 1:
            selected_file_index = st.selectbox("Select image to process:", 
                                         range(len(uploaded_files)), 
                                         format_func=lambda x: uploaded_files[x].name)
            current_file = uploaded_files[selected_file_index]
        else:
            current_file = uploaded_files[0]
    
    st.header("Noise Simulation")
    if 'add_noise' not in st.session_state:
        st.session_state.add_noise = False
        
    add_noise_toggle = st.checkbox("Add Noise (for testing)", value=st.session_state.add_noise)
    
    if add_noise_toggle:
        noise_type = st.selectbox("Noise Type", 
                                 ["gaussian", "salt", "pepper", "s&p", "speckle", "poisson"])
        noise_amount = st.slider("Noise Amount", 0.01, 0.5, 0.05, 0.01)
    
    st.session_state.add_noise = add_noise_toggle
    
    st.header("Processing Methods")
    denoising_method = st.selectbox("Denoising Method", [
        "None",
        "Gaussian Blur", 
        "Median Blur", 
        "Bilateral Filter", 
        "Non-local Means", 
        "Wavelet Denoising", 
        "Total Variation Denoising", 
        "Anisotropic Diffusion"
    ])
    
    enhancement_method = st.selectbox("Enhancement Method", [
        "None", 
        "Histogram Equalization", 
        "CLAHE", 
        "Sharpening", 
        "Super-Resolution",
        "Gamma Correction",
        "Contrast Stretching",
        "Detail Enhancement"
    ])
    
    st.header("Method Parameters")
    params = {}
    
    if denoising_method == "Gaussian Blur" or denoising_method == "Median Blur":
        params["ksize"] = st.slider(f"{denoising_method} Kernel Size", 1, 25, 5, 2)
        
    elif denoising_method == "Bilateral Filter":
        params["d"] = st.slider("Diameter of Pixel Neighborhood", 5, 15, 9, 2)
        params["sigma_color"] = st.slider("Sigma Color", 10, 150, 75, 5)
        params["sigma_space"] = st.slider("Sigma Space", 10, 150, 75, 5)
        
    elif denoising_method == "Non-local Means":
        params["h_param"] = st.slider("Filter Strength (h)", 0.5, 2.0, 1.15, 0.05)
        params["patch_size"] = st.slider("Patch Size", 3, 9, 5, 2)
        params["patch_distance"] = st.slider("Patch Distance", 3, 15, 7, 2)
        
    elif denoising_method == "Wavelet Denoising":
        params["wavelet"] = st.selectbox("Wavelet Type", ["db1", "db2", "haar", "sym2"])
        params["threshold"] = st.slider("Threshold", 0.01, 0.2, 0.04, 0.01)
        
    elif denoising_method == "Total Variation Denoising":
        params["weight"] = st.slider("Weight", 0.01, 0.5, 0.1, 0.01)
        
    elif denoising_method == "Anisotropic Diffusion":
        params["num_iter"] = st.slider("Number of Iterations", 1, 30, 10, 1)
        params["delta_t"] = st.slider("Time Step", 0.01, 0.3, 0.1, 0.01)
        params["kappa"] = st.slider("Kappa", 10, 100, 50, 5)

    if enhancement_method == "CLAHE":
        params["clip_limit"] = st.slider("Clip Limit", 1.0, 5.0, 3.0, 0.5)
        tile_size = st.slider("Tile Grid Size", 2, 16, 8, 2)
        params["tile_size"] = (tile_size, tile_size)
        
    elif enhancement_method == "Sharpening":
        params["strength"] = st.slider("Sharpening Strength", 0.5, 5.0, 1.0, 0.5)
        
    elif enhancement_method == "Super-Resolution":
        params["scale"] = st.slider("Scale Factor", 1.5, 4.0, 2.0, 0.5)
        
    elif enhancement_method == "Gamma Correction":
        params["gamma"] = st.slider("Gamma", 0.1, 3.0, 1.0, 0.1)
        
    elif enhancement_method == "Contrast Stretching":
        params["percentile_low"] = st.slider("Low Percentile", 0, 20, 2, 1)
        params["percentile_high"] = st.slider("High Percentile", 80, 100, 98, 1)
        
    elif enhancement_method == "Detail Enhancement":
        params["sigma_s"] = st.slider("Spatial Sigma", 1, 20, 10, 1)
        params["sigma_r"] = st.slider("Range Sigma", 0.05, 0.45, 0.15, 0.05)
    
    st.header("Output Options")
    output_format = st.selectbox("Output Format", ["PNG", "JPEG", "TIFF"])
    if output_format == "JPEG":
        jpeg_quality = st.slider("JPEG Quality", 0, 100, 90, 5)
        params["jpeg_quality"] = jpeg_quality
    
    output_resize = st.checkbox("Resize Output")
    if output_resize:
        resize_factor = st.slider("Resize Factor", 0.1, 2.0, 1.0, 0.1)
        params["resize_factor"] = resize_factor

if uploaded_files:
    col1, col2 = st.columns(2)
    
    file_bytes = np.asarray(bytearray(current_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    current_file.seek(0)

    if st.session_state.add_noise:
        noisy_image = add_noise(original_image, noise_type, noise_amount)
        display_image = noisy_image.copy()
    else:
        display_image = original_image.copy()

    cache_key = f"{current_file.name}_{denoising_method}_{enhancement_method}_{str(params)}"
    if st.session_state.add_noise:
        cache_key += f"_noise_{noise_type}_{noise_amount}"

    with col1:
        st.subheader("Original Image")
        st.image(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB), use_container_width=True)

    process_button = st.button("Process Image")

    if len(st.session_state.processing_history) > 0:
        history_col1, history_col2 = st.columns(2)
        
        with history_col1:
            if st.button("Undo") and st.session_state.current_state_index > 0:
                st.session_state.current_state_index -= 1
                
        with history_col2:
            if st.button("Redo") and st.session_state.current_state_index < len(st.session_state.processing_history) - 1:
                st.session_state.current_state_index += 1

    result_image = None
    
    if process_button:
        cached_result = get_from_cache(cache_key)
        
        if cached_result is not None:
            result_image = cached_result
            st.success("Retrieved result from cache")
        else:
            with st.spinner("Processing... This may take a few moments"):
                result_queue = queue.Queue()
                processing_thread = threading.Thread(
                    target=process_image,
                    args=(display_image, denoising_method, enhancement_method, params, result_queue)
                )
                processing_thread.start()
                processing_thread.join()
                result = result_queue.get()
                
                if isinstance(result, str) and result.startswith("Error"):
                    st.error(result)
                else:
                    result_image = result
                    save_to_cache(cache_key, result_image)
                    
                    if st.session_state.current_state_index < len(st.session_state.processing_history) - 1:
                        st.session_state.processing_history = st.session_state.processing_history[:st.session_state.current_state_index + 1]
                    
                    st.session_state.processing_history.append({
                        "image": result_image,
                        "denoising": denoising_method,
                        "enhancement": enhancement_method,
                        "params": params.copy(),
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    st.session_state.current_state_index = len(st.session_state.processing_history) - 1

    elif len(st.session_state.processing_history) > 0 and st.session_state.current_state_index >= 0:
        history_item = st.session_state.processing_history[st.session_state.current_state_index]
        result_image = history_item["image"]

        st.info(f"Showing history state {st.session_state.current_state_index + 1}/{len(st.session_state.processing_history)}: "
                f"{history_item['denoising']} + {history_item['enhancement']} at {history_item['timestamp']}")

    if result_image is not None:
        with col2:
            st.subheader("Processed Image")
            st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), use_container_width=True)
            buffer = io.BytesIO()

            if output_resize and "resize_factor" in params:
                download_image = cv2.resize(
                    result_image, 
                    None, 
                    fx=params["resize_factor"], 
                    fy=params["resize_factor"], 
                    interpolation=cv2.INTER_AREA if params["resize_factor"] < 1 else cv2.INTER_CUBIC
                )
            else:
                download_image = result_image

            download_image_rgb = cv2.cvtColor(download_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(download_image_rgb)
            filename = get_filename_without_extension(current_file)
            if output_format == "PNG":
                pil_image.save(buffer, format="PNG")
                mime_type = "image/png"
                download_filename = f"{filename}_processed.png"
            elif output_format == "JPEG":
                quality = params.get("jpeg_quality", 90)
                pil_image.save(buffer, format="JPEG", quality=quality)
                mime_type = "image/jpeg"
                download_filename = f"{filename}_processed.jpg"
            else:  # TIFF
                pil_image.save(buffer, format="TIFF")
                mime_type = "image/tiff"
                download_filename = f"{filename}_processed.tif"
            
            byte_img = buffer.getvalue()
            
            st.download_button(
                label=f"Download as {output_format}",
                data=byte_img,
                file_name=download_filename,
                mime=mime_type
            )

            st.subheader("Image Quality Metrics")
            
            try:
                mse = np.mean((cv2.resize(display_image, (result_image.shape[1], result_image.shape[0])) - result_image) ** 2)
                st.text(f"Mean Squared Error (MSE): {mse:.2f}")
                
                if mse > 0:
                    psnr = 10 * np.log10((255**2) / mse)
                    st.text(f"Peak Signal-to-Noise Ratio (PSNR): {psnr:.2f} dB")
            except:
                st.text("Could not calculate metrics (images may have different dimensions)")

    if len(uploaded_files) > 1:
        st.header("Batch Processing")
        
        if st.button("Process All Images"):
            batch_progress = st.progress(0)
            status_text = st.empty()
            
            batch_results = []
            
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Processing {file.name} ({i+1}/{len(uploaded_files)})")
                
                file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                file.seek(0)

                if st.session_state.add_noise:
                    img = add_noise(img, noise_type, noise_amount)

                result_queue = queue.Queue()
                processing_thread = threading.Thread(
                    target=process_image,
                    args=(img, denoising_method, enhancement_method, params, result_queue)
                )
                processing_thread.start()
                processing_thread.join()
                
                result = result_queue.get()
                if not isinstance(result, str):
                    batch_results.append((file.name, result))

                batch_progress.progress((i + 1) / len(uploaded_files))

            if batch_results:
                status_text.text(f"Processed {len(batch_results)} images successfully!")

                import zipfile
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for filename, img in batch_results:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(img_rgb)
                        img_buffer = io.BytesIO()
                        base_filename = os.path.splitext(filename)[0]
                        
                        if output_format == "PNG":
                            pil_img.save(img_buffer, format="PNG")
                            save_filename = f"{base_filename}_processed.png"
                        elif output_format == "JPEG":
                            quality = params.get("jpeg_quality", 90)
                            pil_img.save(img_buffer, format="JPEG", quality=quality)
                            save_filename = f"{base_filename}_processed.jpg"
                        else: 
                            pil_img.save(img_buffer, format="TIFF")
                            save_filename = f"{base_filename}_processed.tif"
                        
                        zip_file.writestr(save_filename, img_buffer.getvalue())

                st.download_button(
                    label="Download All Processed Images",
                    data=zip_buffer.getvalue(),
                    file_name="processed_images.zip",
                    mime="application/zip"
                )
else:

    st.info("Upload an image to get started! The app will let you apply various denoising and enhancement techniques.")
    
    st.header("Features:")
    features = [
        "💨 Advanced noise reduction using multiple algorithms",
        "✨ Image enhancement and restoration",
        "🎚️ Customizable parameters for each method",
        "📊 Before/after comparison",
        "📂 Batch processing for multiple images",
        "🔄 Undo/redo functionality",
        "📱 Works with various image formats"
    ]
    
    for feature in features:
        st.markdown(feature)