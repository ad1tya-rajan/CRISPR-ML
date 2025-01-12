# CRISPR-ML: Predicting Off-Target Effects with Machine Learning

## Overview
CRISPR-Cas9 is a revolutionary genome-editing tool that allows scientists to make precise edits to DNA. This technology has broad applications in medicine, agriculture, and biotechnology. However, its accuracy is not perfect, and unintended edits ("off-target effects") can occur, leading to potential risks such as undesired mutations or harmful side effects.

This project, **CRISPR-ML**, leverages machine learning to predict off-target effects of CRISPR-Cas9 editing. By analyzing DNA sequences and associated data, this project aims to improve the safety and reliability of CRISPR applications.

## What is CRISPR?
[CRISPR](https://en.wikipedia.org/wiki/CRISPR) (Clustered Regularly Interspaced Short Palindromic Repeats) is a natural system bacteria use to defend against viruses. Scientists have adapted this system for genome editing, combining it with the Cas9 protein to "cut" DNA at specific locations. This is achieved by designing a guide RNA (gRNA) that matches the target DNA sequence.

![CRISPR](images/crispr_img.jpeg)

### Off-Target Effects
While CRISPR-Cas9 is highly specific, it is not perfect. Off-target effects occur when the Cas9 protein cuts DNA at unintended sites, which may:
- Cause harmful mutations.
- Reduce the efficacy of treatments.
- Increase safety concerns in clinical applications.

Identifying and mitigating these off-target effects is a critical challenge in making CRISPR-based therapies safe and reliable in the future.

## Current Implementation
This project includes:
1. **Baseline Models**: Logistic regression and XGBoost models for predicting off-target effects.
2. **Transformer-Based Model**: A modern, transformer-based deep learning architecture designed for sequence data analysis.
3. **Feature Engineering**: One-hot encoding and custom feature extraction.

## Future Directions
- **Cross-Validation and Regularization**: Minimize overfitting in models to improve generalization.
- **Expanded Dataset**: Incorporate diverse datasets to enhance model robustness.
- **Usability**: Extend the project as a Python package or deploy it as a web application for broader accessibility.

## Why This Matters
CRISPR technology is transforming fields like gene therapy, crop engineering, and synthetic biology. However, ensuring its safety is paramount. By addressing the risks of off-target effects with predictive tools, CRISPR-ML contributes to the responsible advancement of genome editing technologies.

## How to Get Involved
If you're interested in contributing to this project or using it in your research, feel free to reach out or explore the codebase.

**Contact:** Aditya Rajan  
**Email:** [rajan9@illinois.edu](mailto:rajan9@illinois.edu)

