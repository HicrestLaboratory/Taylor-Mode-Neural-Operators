# Taylor Mode Neural Operators (TMNO)

This repository contains the official implementation of **Taylor Mode Neural Operators (TMNO)** as presented in the paper:

> **Taylor Mode Neural Operators: Enhancing Computational Efficiency in Physics-Informed Neural Operators**  
> Anonymous, NeurIPS 2024 Machine Learning and the Physical Sciences Workshop.  
> [OpenReview Link](https://openreview.net/forum?id=BvA24ROnJ0)  

## Overview

Taylor Mode Neural Operators introduce a novel application of **Taylor-mode Automatic Differentiation (AD)** to efficiently compute high-order derivatives in **Physics-Informed Neural Operators (PINOs)**. TMNO demonstrates significant computational efficiency in high-order derivative calculations by propagating Taylor coefficients directly through neural operator architectures. The approach is validated on **DeepONet** and **Fourier Neural Operators (FNOs)**, achieving:
- Up to an **order-of-magnitude speed-up** for DeepONet.
- An **eightfold acceleration** for FNOs.

## Code Availability

### Fourier Neural Operators (FNO)
The implementation of TMNO applied to Fourier Neural Operators is publicly available in this repository. Refer to the code and examples in the `fno/` folder for detailed instructions and experiments.

### DeepONet and 3D Extensions
The code for TMNO applied to **DeepONet** and the extension to **3D cases** is available upon request. If interested, please contact us through the repository's issue tracker or via the email provided in the contact section.

## Citation

If you use this repository or its contents in your research, please cite the paper as follows:

@inproceedings{
anonymous2024taylor,
title={Taylor Mode Neural Operators: Enhancing Computational Efficiency in Physics-Informed Neural Operators},
author={Anonymous},
booktitle={Machine Learning and the Physical Sciences Workshop @ NeurIPS 2024},
year={2024},
url={https://openreview.net/forum?id=BvA24ROnJ0}
}




## Contact

For inquiries about the code or additional features, please open an issue or reach out via email (refer to the contact information in the repository).
