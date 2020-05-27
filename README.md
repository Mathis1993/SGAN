# SGAN
Semi-Supervised GAN for predictions based on MRI-slices.

Target: Prediction of subject age (brain age) as a marker for mental disorders.

Idea: In a scenario with many unlabeled but only a few labeld examples, build a GAN learning the distribution behind the data.
A classifier sharing feature layers with the GAN's discriminator predicts the actual labels, while hopefully profiting from
the distribution learned by the GAN.
