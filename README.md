# Intermediate Frame Generator (Work In Progress 🚧)

This project aims to develop an AI-based system for **cluster-aware motion interpolation**, where the goal is to intelligently generate intermediate frames between two given images by understanding and maintaining object boundaries, motion direction, and spatial relationships.

Unlike traditional frame interpolation techniques that rely heavily on optical flow or pixel-based interpolation, this approach focuses on **semantically meaningful color clustering**, convex boundary detection, and projection techniques that imitate 3D light behavior. The architecture is modular, allowing future integration of neural components.

---

## 🚀 Motivation

The goal of this project is to generate **interpolated frames that feel realistic**, especially in cases where multiple objects, lighting variations, or perspective shifts are involved. It’s particularly suited for:

- Low-FPS animations or simulations
- Stylized rendering pipelines
- Synthetic dataset generation
- Creative AI art tools

---

## 🧠 How It Works (Concept Overview)

1. **Color Clustering via HSV Space:**
   - Segments the image into perceptually similar regions using dynamic HSV-based clustering.
   - Each pixel is evaluated in parallel using Python’s `multiprocessing` for speed.

2. **Convex Hull Boundary Extraction:**
   - Uses a custom **gift wrapping algorithm** to compute convex boundaries for each cluster.
   - Ensures clean spatial regions to operate upon in interpolation.

3. **3D Projection-Based Motion Modeling:**
   - Emulates a virtual light source at the center of the original image plane (z = 0).
   - Projects cluster boundaries onto a second virtual image plane (z = d1 + d2), giving a sense of motion and depth.

4. **Cluster Filling & Noise Injection:**
   - After projection, cluster regions are filled with their representative color.
   - A final pass injects controlled noise and fills empty pixels to reduce visual artifacts and add realism.

---

## 🛠 Technologies Used

- Python
- NumPy
- SciPy
- PIL (Pillow)
- Scikit-Image (`skimage.measure`)
- Multiprocessing
- Custom Geometry Algorithms

---

## 📂 Project Status

🔧 **This project is currently in active development.**

The core logic is functional and under validation. Visual outputs are being refined. The next steps include:

- Integrating CNN-based depth or motion priors.
- Smoothing geometric transitions.
- Benchmarking against traditional interpolation tools.

---

## 🧪 How to Run

1. Clone the repository.
2. Place your input image in the `/images` folder and name it `input_image_to_be_analysed.jpg`.
3. Run:

```bash
python main.py
