import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_cameras(extrinsic_matrices):
    """
    Visualizes the extrinsics of multiple cameras in 3D space.

    Parameters:
    - extrinsic_matrices (numpy.ndarray): Array of camera extrinsics matrices (Nx4x4).
    """
    ax = plt.figure().add_subplot(projection='3d')

    for camera_extrinsics in extrinsic_matrices:
        # Extract translation and rotation from camera extrinsics matrix
        translation = camera_extrinsics[:3, 3]
        rotation_matrix = camera_extrinsics[:3, :3]

        # Plot camera position
        ax.scatter(*translation, marker='o')

        # Plot camera orientation axes
        origin = translation
        for i in range(3):
            axis_direction = rotation_matrix[:,i] 
            if i == 0:
                ax.quiver(*origin, *axis_direction, length=0.5, normalize=True)
            else:
                ax.quiver(*origin, *axis_direction, length=1, normalize=True)
        # Plot camera direction
        z = -1 * rotation_matrix[:,2]
        ax.quiver(*origin, *z, length=1, normalize=True, color='r', alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Multiple Cameras Extrinsics Visualization')

    #ax.set_zlim(-2,2)

    plt.show()


# List of extrinsic matrices or poses
# Example poses all pointing at [0,0,0]
poses = np.array([[
    [-0.08952993, -0.98611166,  0.13988634,  5.70651231],
    [ 0.69520015,  0.03870301,  0.71777352, -2.42546075],
    [-0.71321886,  0.16151122,  0.6820799,  18.8415864],
    [ 0.        ,  0.        ,  0.        ,  1.        ]],

   [[-0.10342708,  0.98697917, -0.1231867, -4.71068337],
    [-0.69745921,  0.01633217,  0.71643835,  3.62697491],
    [ 0.70912164,  0.16001682,  0.68668852, 13.1612909],
    [ 0.        ,  0.        ,  0.        ,  1.        ]]])


plot_cameras(poses)