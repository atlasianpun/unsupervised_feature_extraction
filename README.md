# Unsupervised Feature Extraction Project

## Overview

This project implements various unsupervised feature extraction methods to reduce data dimensionality from m to 2 dimensions. The implementation includes different PCA approaches and MDS with exponential scaling.

## Programs

1. **pca1.py**: PCA implementation without mean subtraction
1. **pca2.py**: Standard PCA implementation with mean subtraction
1. **pca3.py**: Normalized PCA implementation with mean subtraction
1. **mds.py**: Multidimensional Scaling with exponentially scaled distances

## Input/Output Format

- **Input**: Comma-separated matrix file (n × m dimensions)
  - Rows represent data points
  - Columns represent features
- **Output**: Comma-separated file containing reduced dimensions (n × 2 matrix)

## Program Details

### PCA Implementations

#### PCA1 (Without Mean Subtraction)

```
python3 pca1.py input_file output_file
```

#### PCA2 (With Mean Subtraction)

```
python3 pca2.py input_file output_file
```

#### PCA3 (Normalized PCA with Mean Subtraction)

```
python3 pca3.py input_file output_file
```

- Computes eigenvectors from the matrix:
C = Σ [(xi − μ)(xi − μ)T / ||xi − μ||^2]
- Summation is over all vectors where xi ≠ μ

### MDS Implementation

```
python3 mds.py input_file output_file alpha
```

- Uses exponentially scaled distances
- Distance formula: distance = (|xi − xj|^2)^(α/2)
- α parameter: 0 ≤ α

## Example Usage

```
python3 pca1.py irises.data irises2d.datapython3 pca2.py irises.data irises2d.datapython3 pca3.py irises.data irises2d.datapython3 mds.py irises.data irises2d.data 0.1
```



## Project Requirements

### Implementation Guidelines

- Each program must run independently
- No code sharing between programs
- Programs must run from command line
- Use only common Python libraries
- Follow specified input/output formats

### Research Components

1. Compare PCA implementations:

- Analyze differences between standard (pca2) and normalized (pca3) PCA
- Identify scenarios for using each approach

1. MDS Analysis:

- Study effects of α parameter
- Document recommendations for α values in different scenarios

## Repo Files

   1. Python scripts:
  - pca1.py
  - pca2.py
  - pca3.py
  - mds.py
   1. Documentation
   1. Research report

