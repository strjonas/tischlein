"""Tool orientation error, including 180-degree and jaw-alignment cases."""
import numpy as np
from scipy.spatial.transform import Rotation


def tool_error(R, approach, jaw=None):
    x = np.asarray(approach, float)
    x /= np.linalg.norm(x)
    if jaw is not None:
        y = np.asarray(jaw, float)-x*np.dot(x, jaw)
        y /= np.linalg.norm(y)
        if y@R[:, 1] < 0:
            y = -y
        target = np.column_stack([x, y, np.cross(x, y)])
        return Rotation.from_matrix(target@R.T).as_rotvec()
    cross = np.cross(R[:, 0], x)
    norm = np.linalg.norm(cross)
    dot = float(np.clip(R[:, 0]@x, -1, 1))
    if norm < 1e-8:
        return np.pi*R[:, 1] if dot < 0 else np.zeros(3)
    return cross/norm*np.arctan2(norm, dot)
