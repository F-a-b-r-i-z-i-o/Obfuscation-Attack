# Obfuscation-Attack

Adversarial machine learning is a series of techniques aimed at compromising the correct functioning of a computer system that makes use of machine learning algorithms, through the construction of special inputs capable of deceiving such algorithms: more specifically, the purpose of these techniques is to cause misclassification in one of these algorithms. 


The goal of an obfuscation (aka evasion) attack is to violate the integrity of a machine learning model.
During an obfuscation attack, the attacker modifies a certain sample with the aim of obtaining as output from the classifier a class that is different from its real class of belonging; alternatively, the attack could more simply attempt to decrease the confidence of the model for that sample. 

the code is divided into 3 parts.

1. part: 
concerns instagram filters created using opencv.

2. part:
all the functions that modify the various parameters of the filters.

3. part:
how images and filter parameters are taken as input.

In this version an input folder is passed and all the photos are modified with different intensity and alpha values for each photo. So the same for the same photo. Alpha and intensity are randomly taken only for the first photo.

Python 3.8.5 

Opencv 4.5.3
