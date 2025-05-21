import numpy as np
import matplotlib.pyplot as plt
import struct
import cmocean  # modern colormaps

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12
})

def read_fortran_2d_array(filename):
    with open(filename, 'rb') as f:
        f.read(4)
        ndim = struct.unpack('i', f.read(4))[0]
        f.read(4)

        f.read(4)
        if ndim == 2:
            nx, ny = struct.unpack('ii', f.read(8))
        else:
            nx, ny, *_ = struct.unpack('iiii', f.read(16))
        f.read(4)

        f.read(4)
        data = np.fromfile(f, dtype=np.float32, count=nx * ny)
        f.read(4)

        return data.reshape((nx, ny), order='F')

def main():
    I = read_fortran_2d_array('input/test_Kolm_L512V_L512_I')
    Q = read_fortran_2d_array('input/test_Kolm_L512V_L512_Q')
    U = read_fortran_2d_array('input/test_Kolm_L512V_L512_U')
    ang_file = read_fortran_2d_array('input/test_Kolm_L512V_L512_ang')

    angle_computed = 0.5 * np.arctan2(U, Q)
    angle_computed_deg = np.degrees(angle_computed)
    ang_file_deg = np.degrees(ang_file) if np.max(np.abs(ang_file)) <= np.pi else ang_file

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    im0 = axs[0, 0].imshow(I.T, origin='lower', cmap='cividis')
    axs[0, 0].set_title('Stokes I (Intensity)')
    plt.colorbar(im0, ax=axs[0, 0], fraction=0.046)

    im1 = axs[0, 1].imshow(Q.T, origin='lower', cmap='coolwarm', vmin=-np.max(np.abs(Q)), vmax=np.max(np.abs(Q)))
    axs[0, 1].set_title('Stokes Q')
    plt.colorbar(im1, ax=axs[0, 1], fraction=0.046)

    im2 = axs[1, 0].imshow(U.T, origin='lower', cmap='coolwarm', vmin=-np.max(np.abs(U)), vmax=np.max(np.abs(U)))
    axs[1, 0].set_title('Stokes U')
    plt.colorbar(im2, ax=axs[1, 0], fraction=0.046)

    im3 = axs[1, 1].imshow(angle_computed_deg.T, origin='lower', cmap='twilight_shifted', vmin=-90, vmax=90)
    axs[1, 1].set_title('Polarization Angle (from Q/U)')
    plt.colorbar(im3, ax=axs[1, 1], fraction=0.046)

    for ax in axs.flat:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    plt.suptitle('Polarization Maps', fontsize=16, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == "__main__":
    main()
