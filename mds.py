import numpy as np
import pandas as pd
import sys
from scipy.spatial.distance import pdist, squareform


def check_input(input_file):
    try:
        data = pd.read_csv(input_file, header=None)
        if data.shape[1] < 2:
            print(f"Error: Input data should have at least 2 columns. Found {data.shape[1]} column(s).")
            return None
        if data.shape[0] < 2:
            print(f"Error: Input data should have at least 2 rows. Found {data.shape[0]} row(s).")
            return None
        if not data.applymap(np.isreal).all().all():
            print("Error: Input data contains non-numeric values.")
            return None
        print(f"Input data shape: {data.shape[0]} rows × {data.shape[1]} columns")
        return data.values
    except Exception as e:
        print(f"Error: Unable to read the input file as a valid CSV. {str(e)}")
        return None


def mds_exponential(data, n_components=2, alpha=1.0):
  # Compute pairwise squared distances
  squared_distances = pdist(data, 'sqeuclidean')

  # Apply the custom scaling
  scaled_distances = np.power(squared_distances, alpha / 2)

  # Convert to square form
  D = squareform(scaled_distances)

  # Double centering
  n = D.shape[0]
  H = np.eye(n) - np.ones((n, n)) / n
  B = -0.5 * H.dot(D ** 2).dot(H)

  # Eigendecomposition
  eigenvalues, eigenvectors = np.linalg.eigh(B)

  # Sort eigenvalues in descending order
  idx = np.argsort(eigenvalues)[::-1]
  eigenvalues = eigenvalues[idx]
  eigenvectors = eigenvectors[:, idx]

  # Select top n_components
  eigenvalues = eigenvalues[:n_components]
  eigenvectors = eigenvectors[:, :n_components]

  # Compute the coordinates
  coordinates = eigenvectors.dot(np.diag(np.sqrt(np.abs(eigenvalues))))

  return coordinates


def main(input_file, output_file, alpha):
    data = check_input(input_file)
    if data is None:
        return

    reduced_data = mds_exponential(data, alpha=alpha)
    pd.DataFrame(reduced_data).to_csv(output_file, header=False, index=False)
    print(f"Reduced data saved to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python mds.py <input_file> <output_file> <alpha>")
    else:
        input_file, output_file = sys.argv[1], sys.argv[2]
        alpha = float(sys.argv[3])
        main(input_file, output_file, alpha)

# Created/Modified files during execution:
print(output_file)