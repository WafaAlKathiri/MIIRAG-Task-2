# MIIRAG-Task-2

Cracks in road surfaces may seem minor, but if left undetected, they can lead to costly repairs and serious safety hazards. Traditional crack inspection methods are slow, labor-intensive, and prone to errors, while existing automated systems struggle with inconsistent data and complex crack patterns. To bridge this gap, our project leverages Generative AI to create synthetic yet realistic crack images, improving dataset quality for Instance Segmentation—a hybrid approach combining Object Detection and Semantic Segmentation. By enriching training data, we aim to solve data imbalance issues in exsisting road crack image datasets, ensuring more precise and scalable crack detection for smarter infrastructure maintenance.

1.	Data collection and Cleaning 
This project leveraged three datasets for road crack detection: EdmCrack600, China-Data (RDD2022-China subset), and the Road Crack Detection dataset from Roboflow Universe.
•	EdmCrack600: (600 images) Accessible via direct download, this Canadian dataset provides high-resolution images with pixel-level annotations (boxes and segmentation). Split: 480 train, 60 valid, 60 test. All images from this dataset were used. 
•	China-Data (RDD2022-China subset): (2649 images) Publicly available, this subset contains Chinese road images with annotations for diverse crack types. Selected for its variety, data points, and box ground truth to enhance results. Split: 1900 train, 527 valid, 266 test. Some images from this dataset were used to improve training of minority classes.
•	Road Crack Detection (Roboflow): (1,579 images) Accessed via Roboflow, this dataset expands the training data with additional images and bounding box annotations. The Roboflow data was chosen to further expand the diversity of training data. Split: 1105 train, 314 valid, 160 test. Some images from this dataset were used. 

In terms of cleaning the data, the images in the datasets consisted of single images that contained multiple cracks and their types. There are 8 classes of cracks: alligator crack (class 0), block crack (class 1), diagonal crack (class 2), edge crack (class 3), longitudinal crack (class 4), pothole (class 5), reflective crack (class 6), transverse crack (class 7). Therefore, to process each type of crack separatly we created a program to separate the cracks in each image based on the labels (provided with the datasets) and save them in different folders according to their class.The images were then preprocessed to standardize image resolution (512x512) and normalize pixel values for compatibility with deep learning models. Other traditional augmentation techniques were not utilized as the various crack types might make it tricky (for example, rotating a longitudinal crack image might tranform it to a diagonal crack). 

NOTE : Only some images from the RDD2022-China and Road Crack Detection (Roboflow) datasets were used to improve training results of minority classes (crack Types:  0, 1 , 2, 3 and 5). The main dataset we depended on in this project was the EdmCrack600 and it is the only one where we used all of the images in it. 


2.	Generating Synthetic Road Crack Images - Comparative Study 
To address the data imbalance we aimed to generate high-quality synthetic crack images, by evaluating three GAN model architectures. Generative Adversarial Networks (GANs) are a class of machine learning models consisting of two neural networks—a generator and a discriminator—that compete in a zero-sum game to produce highly realistic synthetic data. The generator creates fake samples, while the discriminator distinguishes real from fake, improving both networks through adversarial training until the generated data becomes indistinguishable from real data (Goodfellow et al., 2014). We utilized notebooks and GPU cores on Kaggle to run the training process with Python, as well as Pytorch, Tensorflow and Numpy libraries for the following architectures:

•	StyleGAN2
•	Conditional GAN (CGAN)
•	BigBiGAN

To ensure a fair comparison, we trained all models on Class 4 of the EDM600 dataset with equal hyper-parameters (Adam optimizor, same number of training hours, batch size, epochs and image size). Each model was assessed based on its ability to generate realistic and structurally accurate crack patterns.


