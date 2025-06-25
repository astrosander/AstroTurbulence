import h5py

# File path
filename = 'ms01ma08.mhd_w.00300.vtk.h5'

# Open the HDF5 file in read mode
with h5py.File(filename, 'r') as f:
    # Print all root-level keys (datasets or groups)
    print("Root keys:")
    for key in f.keys():
        print(f"  {key}")

    # Example: explore one group/dataset
    group_name = list(f.keys())[0]  # change as needed
    print(f"\nExploring '{group_name}':")
    data = f[group_name]

    # If it's a dataset, show shape and dtype
    if isinstance(data, h5py.Dataset):
        print(f"Dataset shape: {data.shape}")
        print(f"Data type: {data.dtype}")
        print(f"First few values: {data[...][:5]}")
    else:
        print("This is a group. Contains:")
        for subkey in data.keys():
            print(f"  {subkey}")
