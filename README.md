**This is the technical report of experiment applying knowledge distillation on classification task.** (Submitted on computer vision class)

# Datasets
- CIFAR10

# Models
- Teacher model : Resnet152
- Student model : Resnet18

# Techniques
-  Augmentation
    - Albumentation
    - Transforms
    - Test Time Augmentation (TTA)

- Loss function
    - Mean Squared Error (Soft/Hard)
    - Cross Entropy
    - KL divergence

# Experiments
1. Finding best combination of hyperparameters for integral loss
2. Finding the differences between join_train method and each_train method with different loss functions
3. Finding the losses when the different TTA is applied
4. Finding the losses when the different augmentation is applied -> if possible
