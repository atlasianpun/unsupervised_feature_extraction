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

def pca_with_mean_subtraction(data, n_components=2):
  mean_centered_data = data - np.mean(data, axis=0)
  covariance_matrix = np.cov(mean_centered_data.T)
  eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
  idx = np.argsort(eigenvalues)[::-1]
  eigenvectors = eigenvectors[:, idx]
  selected_eigenvectors = eigenvectors[:, :n_components]
  reduced_data = np.dot(mean_centered_data, selected_eigenvectors)
  return reduced_data

def main(input_file, output_file):
  data = check_input(input_file)
  if data is None:
      return
  reduced_data = pca_with_mean_subtraction(data)
  pd.DataFrame(reduced_data).to_csv(output_file, header=False, index=False)
  print(f"Reduced data saved to {output_file}")

if __name__ == "__main__":
  if len(sys.argv) != 3:
      print("Usage: python pca2.py <input_file> <output_file>")
  else:
      input_file, output_file = sys.argv[1], sys.argv[2]
      main(input_file, output_file)

# Created/Modified files during execution:
print(output_file)