"""Independent geometry checks before simulator data reaches the model."""
import unittest
import numpy as np
from prepare_capture import project_depth, rigid


class GeometryTests(unittest.TestCase):
    def test_occlusion_and_optical_depth(self):
        # Both points hit pixel (4,3); optical-axis depth is 2, not range.
        points = np.array([[1., 0, 2], [2., 0, 4], [0, 0, -2], [100, 0, 1]])
        intrinsic = np.array([[2., 0, 3], [0, 2., 3], [0, 0, 1.]])
        depth = project_depth(points, np.eye(4), intrinsic, 8, 8)
        self.assertEqual(depth[3, 4], 2.)
        self.assertEqual(np.count_nonzero(depth), 1)

    def test_translated_camera(self):
        optical_from_lidar = np.eye(4)
        optical_from_lidar[0, 3] = -1
        depth = project_depth(np.array([[1., 0, 3.]]), optical_from_lidar, np.eye(3), 4, 4)
        self.assertEqual(depth[0, 0], 3.)

    def test_reflection_is_not_a_rigid_rotation(self):
        with self.assertRaises(ValueError):
            rigid(np.diag([-1., 1, 1, 1]))


if __name__ == '__main__':
    unittest.main()
