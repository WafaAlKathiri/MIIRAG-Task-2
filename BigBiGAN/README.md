BigBiGAN, developed by DeepMind, is a powerful Generative Adversarial Network (GAN) that excels at unsupervised image generation and representation learning. Unlike traditional GANs, BigBiGAN not only generates high-quality images but also learns meaningful latent representations, making it highly adaptable for various image synthesis tasks. We found that Fine-tuning and training BigBiGAN to generate realistic road crack images was a promising approach because it can capture the intricate textures and irregular patterns of different crack types, addressing data scarcity and class imbalance. Nevertheless, the pretrained model’s architicture turned out to be very complex which resulted in our inability to use GPU capabilities while attempting to fine-tune it. As a result, trining this model did not generate any initial (synthetic image) results within the same number of training hours as StyleGan2 and CGAN did. Hence it was not our choice of GAN model to work with. 

The notebook was created and used on Kaggle, which is where the original model file was found. To access the model file, after creating a new notebook on Kaggle: 
1. Go to Input
2. Press "Add input"
3. In the search box type BigBiGAN
4. You should see an option with Google DeepMind as the creater
5. Press the "+" sign next to it to add it to your notebook

Model page link on Kaggle : https://www.kaggle.com/models/deepmind/bigbigan 
