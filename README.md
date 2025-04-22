# Advanced Image Restoration and Denoising App

## Overview

This application provides a comprehensive set of image restoration and enhancement tools through an interactive Streamlit interface. It combines various denoising algorithms with image enhancement techniques to repair and improve digital images.

![App Screenshot](https://via.placeholder.com/800x400?text=Image+Restoration+App)

## Features

- **Advanced Denoising Algorithms**:
  - Gaussian Blur
  - Median Blur
  - Bilateral Filter
  - Non-local Means
  - Wavelet Denoising
  - Total Variation Denoising
  - Anisotropic Diffusion

- **Enhancement Techniques**:
  - Histogram Equalization
  - CLAHE (Contrast Limited Adaptive Histogram Equalization)
  - Sharpening
  - Super-Resolution
  - Gamma Correction
  - Contrast Stretching
  - Detail Enhancement

- **Additional Features**:
  - Noise simulation for testing
  - Customizable parameters for each method
  - Real-time before/after comparison
  - Batch processing for multiple images
  - Undo/redo functionality
  - Multiple output formats (PNG, JPEG, TIFF)
  - Image quality metrics (MSE, PSNR)

## Installation

### Prerequisites

- Python 3.7+

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/shauryarannot/Image-Restoration-and Colorization.git
   cd Image-Restoration
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   streamlit run app.py
   ```

## Usage Guide

1. **Upload Images**: 
   - Use the file uploader to select one or more images
   - For multiple images, you can select which one to process individually

2. **Optional Noise Simulation**:
   - Add synthetic noise to test denoising algorithms
   - Adjust noise type and amount

3. **Select Processing Methods**:
   - Choose a denoising algorithm
   - Choose an enhancement technique
   - Adjust parameters for fine-tuning

4. **Process and Download**:
   - Click "Process Image" to apply selected methods
   - Download the processed image in your preferred format
   - For batch processing, use the "Process All Images" button

5. **History Navigation**:
   - Use Undo/Redo buttons to navigate through processing history

## Technical Details

The application leverages several libraries for image processing:

- **OpenCV**: Fast image processing operations
- **scikit-image**: Advanced image restoration algorithms
- **PyWavelets**: Wavelet-based denoising
- **NumPy**: Numerical operations on image arrays
- **Pillow**: Image format conversion and saving
- **Streamlit**: Web application interface

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [OpenCV](https://opencv.org/)
- [scikit-image](https://scikit-image.org/)
- [Streamlit](https://streamlit.io/)
