"""Transform dùng quaternion wxyz; không phụ thuộc SciPy/MuJoCo."""
import numpy as np


def normalize_quaternion(q):
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.all(np.isfinite(q)) or np.any(norm < 1e-8):
        raise ValueError("Quaternion không hợp lệ")
    return q / norm


def continuous_quaternions(q):
    result = normalize_quaternion(q).copy()
    if result[0, 0] < 0:
        result[0] *= -1
    for i in range(1, len(result)):
        if np.dot(result[i-1], result[i]) < 0:
            result[i] *= -1
    return result


def pose_matrix(pose):
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        raise ValueError("Pose phải là vector hữu hạn 7D")
    w, x, y, z = normalize_quaternion(pose[3:])
    out = np.eye(4)
    out[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                   [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                   [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    out[:3, 3] = pose[:3]
    return out


def rotvec_matrix(pose):
    pose = np.asarray(pose, dtype=np.float64)
    theta = np.linalg.norm(pose[3:])
    out = np.eye(4)
    if theta > 1e-12:
        x, y, z = pose[3:] / theta
        skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        out[:3, :3] += np.sin(theta) * skew + (1-np.cos(theta)) * (skew @ skew)
    out[:3, 3] = pose[:3]
    return out


def check_transform(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("Transform không hợp lệ")
    r = matrix[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(r), 1, atol=1e-5):
        raise ValueError("Rotation không thuộc SO(3)")
    if not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise ValueError("Dòng cuối transform sai")
    return matrix
