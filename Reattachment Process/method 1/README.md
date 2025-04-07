# Road Crack Reattachment Pipeline

The file reattachment1.py provides a full pipeline to reattach synthetic crack images onto real road backgrounds from the dataset EdmCrack600 using a blend of image similarity, enhancement, and Poisson blending techniques. 

---

##  Method Overview

The system works in 3 main steps:

### 1. **Best Match Search**
- Computes **color + texture similarity** between each crack image and road image.
- Uses **Lab color histograms** and **SSIM** to evaluate visual compatibility.
- Selects the best matching road background for each crack.

### 2. **Crack Placement & Harmonization**
- Searches for a **clean region** on the road mask to place the crack (avoiding overlap with existing cracks).
- Applies:
  - **Histogram Matching** for color harmonization.
  - **Lighting Adjustment** to blend lighting.
  - **Noise Addition** for realism.
  - **Poisson Blending** for seamless integration.

### 3. **Batch Processing**
- Automatically processes all crack images in a folder and saves the reattached results.
