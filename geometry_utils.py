import numpy as np
import pandas as pd


def rotate_points_around_y(df: pd.DataFrame, angle_deg: float, origin=(0.0, 0.0, 0.0)) -> pd.DataFrame:
    """
    Rotate all points around the global Y-axis by angle_deg degrees.

    Args:
        df: DataFrame with columns ['X','Y','Z'] (and optionally 'Node Number']).
        angle_deg: Rotation angle in degrees (right-hand rule about +Y).
        origin: Pivot point (x0, y0, z0) to rotate about. Defaults to the global origin.

    Returns:
        A new DataFrame with rotated coordinates.
    """
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    # Rotation about Y: Ry = [[c,0,s],[0,1,0],[-s,0,c]]
    R = np.array([[c, 0.0, s],
                  [0.0, 1.0, 0.0],
                  [-s, 0.0, c]])

    coords = df[['X', 'Y', 'Z']].to_numpy(dtype=float)
    pivot = np.array(origin, dtype=float)
    rotated = (coords - pivot) @ R.T + pivot

    out = df.copy()
    out[['X', 'Y', 'Z']] = rotated
    return out
