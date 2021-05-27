# Automated Artefacts Detection (Tensorflow Python)
Automated artefacts detection for OCT-Angiography images using deep learning image classification (CNN). The goal of this work is to develop an OCT-Angiography based image Defect Detection System (DDS) using deep learning to automatically identify the presence of artefacts in OCT-Angiography images in order to reduce misinterpretation.

![Screenshot](idea.png)

# Labelling Mechanisms

![Screenshot](label.png)

# Results
A total of 420 patches (210 normal patches and 210 defect patches) will be used for network training and validation. 5-fold cross validation wasperformed to evaluate the accuracy. Table I shows the overall accuracy for the three networks.

![Screenshot](table1.png)

# Integrate UI with .H5 model
Left: OCT-Angiography image with artefacts. Right: Final outcome of the image where the red patches contain defects while the green patches indicates does not contain any defects.

![Screenshot](final.png)
