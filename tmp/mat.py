import scipy.io as sio

# Load the .mat file
mat_data = sio.loadmat('F:/reproduction/dataset/PRW-v16.04.20/annotations/c1s1_000151.jpg.mat')

# Print the keys to understand the structure
print(mat_data.keys())

# Access specific data
# For example, if there's a key named 'data', you can access it like this:
data = mat_data['box_new']
print(data)
