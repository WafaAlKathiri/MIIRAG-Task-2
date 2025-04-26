This folder contains the notebook used to implement the GAN texture transfer method. 

Instead of generating the synthetic crack separately then placing it on top of the original real image, this method generates the synthetic road crack image directly on empty spots of the real road crack image. After that it manipulates the pixels to match the color of original spot and then blend using poison blending. 

The only issue is that the synthetic road crack images are always generated with a white border arond the edges, so this method also attempts to remove them after generation and before blending. As can be seen in the image, the method worked with some cracks but not all, two in the bottom left seem to be blended quite well, but the rest need imrovement.  
