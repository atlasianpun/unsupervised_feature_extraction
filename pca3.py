import numpy as np
import pandas as pd
import sys

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

def normalized_pca_with_mean_subtraction(data, n_components=2):
    mu = np.mean(data, axis=0)

    # Center the data
    centered_data = data - mu

    # Initialize the covariance matrix
    C = np.zeros((data.shape[1], data.shape[1]))

    # Compute the normalized covariance matrix
    for x in centered_data:
        norm_squared = np.linalg.norm(x) ** 2
        C += np.outer(x, x) / norm_squared

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(C)

    # Sort eigenvectors by eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Select the top n_components eigenvectors
    selected_eigenvectors = eigenvectors[:, :n_components]

    # Project the data onto the new feature space
    reduced_data = np.dot(centered_data, selected_eigenvectors)
    return reduced_data

def main(input_file, output_file):
  data = check_input(input_file)
  if data is None:
      return
  reduced_data = normalized_pca_with_mean_subtraction(data)
  pd.DataFrame(reduced_data).to_csv(output_file, header=False, index=False)
  print(f"Reduced data saved to {output_file}")

if __name__ == "__main__":
  if len(sys.argv) != 3:
      print("Usage: python pca3.py <input_file> <output_file>")
  else:
      input_file, output_file = sys.argv[1], sys.argv[2]
      main(input_file, output_file)

# Created/Modified files during execution:
print(output_file)