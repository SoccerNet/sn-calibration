import argparse
import json
import os

import cv2 as cv
import numpy as np
from SoccerNet.utils import getListGames
from tqdm import tqdm

from src.soccerpitch import SoccerField


def pan_tilt_roll_to_orientation(pan, tilt, roll):
    """
    Conversion from euler angles to orientation matrix.
    :param pan:
    :param tilt:
    :param roll:
    :return: orientation matrix
    """
    Rpan = np.array([
        [np.cos(pan), -np.sin(pan), 0],
        [np.sin(pan), np.cos(pan), 0],
        [0, 0, 1]])
    Rroll = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]])
    Rtilt = np.array([
        [1, 0, 0],
        [0, np.cos(tilt), -np.sin(tilt)],
        [0, np.sin(tilt), np.cos(tilt)]])
    rotMat = np.dot(Rpan, np.dot(Rtilt, Rroll))
    return rotMat


def rotation_matrix_to_pan_tilt_roll(rotation):
    """
    Decomposes the rotation matrix into pan, tilt and roll angles. There are two solutions, but as we know that cameramen
    try to minimize roll, we take the solution with the smallest roll.
    :param rotation: rotation matrix
    :return: pan, tilt and roll in radians
    """
    orientation = np.transpose(rotation)
    first_tilt = np.arccos(orientation[2, 2])
    second_tilt = - first_tilt

    sign_first_tilt = 1. if np.sin(first_tilt) > 0. else -1.
    sign_second_tilt = 1. if np.sin(second_tilt) > 0. else -1.

    first_pan = np.arctan2(sign_first_tilt * orientation[0, 2], sign_first_tilt * - orientation[1, 2])
    second_pan = np.arctan2(sign_second_tilt * orientation[0, 2], sign_second_tilt * - orientation[1, 2])
    first_roll = np.arctan2(sign_first_tilt * orientation[2, 0], sign_first_tilt * orientation[2, 1])
    second_roll = np.arctan2(sign_second_tilt * orientation[2, 0], sign_second_tilt * orientation[2, 1])

    # print(f"first solution {first_pan*180./np.pi}, {first_tilt*180./np.pi}, {first_roll*180./np.pi}")
    # print(f"second solution {second_pan*180./np.pi}, {second_tilt*180./np.pi}, {second_roll*180./np.pi}")
    if np.fabs(first_roll) < np.fabs(second_roll):
        return first_pan, first_tilt, first_roll
    return second_pan, second_tilt, second_roll


def unproject_image_point(homography, point2D):
    """
    Given the homography from the world plane of the pitch and the image and a point localized on the pitch plane in the
    image, returns the coordinates of the point in the 3D pitch plane.
    /!\ Only works for correspondences on the pitch (Z = 0).
    :param homography: the homography
    :param point2D: the image point whose relative coordinates on the world plane of the pitch are to be found
    :return: A 2D point on the world pitch plane in homogenous coordinates (X,Y,1) with X and Y being the world
    coordinates of the point.
    """
    hinv = np.linalg.inv(homography)
    pitchpoint = hinv @ point2D
    pitchpoint = pitchpoint / pitchpoint[2]
    return pitchpoint


class Camera:

    def __init__(self, iwidth=960, iheight=540):
        self.position = np.zeros(3)
        self.rotation = np.eye(3)
        self.calibration = np.eye(3)
        self.radial_distortion = np.zeros(6)
        self.thin_prism_disto = np.zeros(4)
        self.tangential_disto = np.zeros(2)
        self.image_width = iwidth
        self.image_height = iheight
        self.xfocal_length = 1
        self.yfocal_length = 1
        self.principal_point = (self.image_width / 2, self.image_height / 2)

    def solve_pnp(self, point_matches):
        """
        With a known calibration matrix, this method can be used in order to retrieve rotation and translation camera
        parameters.
        :param point_matches: A list of pairs of 3D-2D point matches .
        """
        target_pts = np.array([pt[0] for pt in point_matches])
        src_pts = np.array([pt[1] for pt in point_matches])
        _, rvec, t, inliers = cv.solvePnPRansac(target_pts, src_pts, self.calibration, None)
        self.rotation, _ = cv.Rodrigues(rvec)
        self.position = - np.transpose(self.rotation) @ t.flatten()

    def refine_camera(self, pointMatches):
        """
        Once that there is a minimal set of initial camera parameters (calibration, rotation and position roughly known),
        this method can be used to refine the solution using a non-linear optimization procedure.
        :param pointMatches:  A list of pairs of 3D-2D point matches .

        """
        rvec, _ = cv.Rodrigues(self.rotation)
        target_pts = np.array([pt[0] for pt in pointMatches])
        src_pts = np.array([pt[1] for pt in pointMatches])

        rvec, t = cv.solvePnPRefineLM(target_pts, src_pts, self.calibration, None, rvec, -self.rotation @ self.position,
                                      (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 20000, 0.00001))
        self.rotation, _ = cv.Rodrigues(rvec)
        self.position = - np.transpose(self.rotation) @ t

    def from_homography(self, homography):
        """
        This method initializes the essential camera parameters from the homography between the world plane of the pitch
        and the image. It is based on the extraction of the calibration matrix from the homography (Algorithm 8.2 of
        Multiple View Geometry in computer vision, p225), then using the relation between the camera parameters and the
        same homography, we extract rough rotation and position estimates (Example 8.1 of Multiple View Geometry in
        computer vision, p196).
        :param homography: The homography that captures the transformation between the 3D flat model of the soccer pitch
         and its image.
        """
        success, _ = self.estimate_calibration_matrix_from_plane_homography(homography)
        if not success:
            return False

        hprim = np.linalg.inv(self.calibration) @ homography
        lambda1 = 1 / np.linalg.norm(hprim[:, 0])
        lambda2 = 1 / np.linalg.norm(hprim[:, 1])
        lambda3 = np.sqrt(lambda1 * lambda2)

        r0 = hprim[:, 0] / lambda1
        r1 = hprim[:, 1] / lambda2
        r2 = np.cross(r0, r1)

        R = np.column_stack((r0, r1, r2))
        u, s, vh = np.linalg.svd(R)
        R = u @ vh
        if np.linalg.det(R) < 0:
            u[:, 2] *= -1
            R = u @ vh
        self.rotation = R
        t = hprim[:, 2] * lambda3
        self.position = - np.transpose(R) @ t
        return True

    def to_json_parameters(self):
        """
        Saves camera to a JSON serializable dictionary.
        :return: The dictionary
        """
        pan, tilt, roll = rotation_matrix_to_pan_tilt_roll(self.rotation)
        camera_dict = {
            "pan_degrees": pan * 180. / np.pi,
            "tilt_degrees": tilt * 180. / np.pi,
            "roll_degrees": roll * 180. / np.pi,
            "position_meters": self.position.tolist(),
            "x_focal_length": self.xfocal_length,
            "y_focal_length": self.yfocal_length,
            "principal_point": [self.principal_point[0], self.principal_point[1]],
            "radial_distortion": self.radial_distortion.tolist(),
            "tangential_distortion": self.tangential_disto.tolist(),
            "thin_prism_distortion": self.thin_prism_disto.tolist()

        }
        return camera_dict

    def from_json_parameters(self, calib_json_object):
        """
        Loads camera parameters from dictionary.
        :param calib_json_object: the dictionary containing camera parameters.
        """
        self.principal_point = calib_json_object["principal_point"]
        self.image_width = 2 * self.principal_point[0]
        self.image_height = 2 * self.principal_point[1]
        self.xfocal_length = calib_json_object["x_focal_length"]
        self.yfocal_length = calib_json_object["y_focal_length"]

        self.calibration = np.array([
            [self.xfocal_length, 0, self.principal_point[0]],
            [0, self.yfocal_length, self.principal_point[1]],
            [0, 0, 1]
        ], dtype='float')

        pan = calib_json_object['pan_degrees'] * np.pi / 180.
        tilt = calib_json_object['tilt_degrees'] * np.pi / 180.
        roll = calib_json_object['roll_degrees'] * np.pi / 180.

        self.rotation = np.array([
            [-np.sin(pan) * np.sin(roll) * np.cos(tilt) + np.cos(pan) * np.cos(roll),
             np.sin(pan) * np.cos(roll) + np.sin(roll) * np.cos(pan) * np.cos(tilt), np.sin(roll) * np.sin(tilt)],
            [-np.sin(pan) * np.cos(roll) * np.cos(tilt) - np.sin(roll) * np.cos(pan),
             -np.sin(pan) * np.sin(roll) + np.cos(pan) * np.cos(roll) * np.cos(tilt), np.sin(tilt) * np.cos(roll)],
            [np.sin(pan) * np.sin(tilt), -np.sin(tilt) * np.cos(pan), np.cos(tilt)]
        ], dtype='float')

        self.rotation = np.transpose(pan_tilt_roll_to_orientation(pan, tilt, roll))

        self.position = np.array(calib_json_object['position_meters'], dtype='float')

        self.radial_distortion = np.array(calib_json_object['radial_distortion'], dtype='float')
        self.tangential_disto = np.array(calib_json_object['tangential_distortion'], dtype='float')
        self.thin_prism_disto = np.array(calib_json_object['thin_prism_distortion'], dtype='float')

    def distort(self, point):
        """
        Given a point in the normalized image plane, apply distortion
        :param point: 2D point on the normalized image plane
        :return: 2D distorted point
        """
        numerator = 1
        denominator = 1
        radius = np.sqrt(point[0] * point[0] + point[1] * point[1])

        for i in range(3):
            k = self.radial_distortion[i]
            numerator += k * radius ** (2 * (i + 1))
            k2n = self.radial_distortion[i + 3]
            denominator += k2n * radius ** (2 * (i + 1))

        radial_distortion_factor = numerator / denominator
        xpp = point[0] * radial_distortion_factor + \
              2 * self.tangential_disto[0] * point[0] * point[1] + self.tangential_disto[1] * (
                      radius ** 2 + 2 * point[0] ** 2) + \
              self.thin_prism_disto[0] * radius ** 2 + self.thin_prism_disto[1] * radius ** 4
        ypp = point[1] * radial_distortion_factor + \
              2 * self.tangential_disto[1] * point[0] * point[1] + self.tangential_disto[0] * (
                      radius ** 2 + 2 * point[1] ** 2) + \
              self.thin_prism_disto[2] * radius ** 2 + self.thin_prism_disto[3] * radius ** 4
        return np.array([xpp, ypp], dtype=np.float32)

    def project_point(self, point3D, distort=True):
        """
        Uses current camera parameters to predict where a 3D point is seen by the camera.
        :param point3D: The 3D point in world coordinates.
        :param distort: optional parameter to allow projection without distortion.
        :return: The 2D coordinates of the imaged point
        """
        point = point3D - self.position
        rotated_point = self.rotation @ np.transpose(point)
        if rotated_point[2] < 0.:
            return np.zeros(3)
        rotated_point = rotated_point / rotated_point[2]
        if distort:
            distorted_point = self.distort(rotated_point)
        else:
            distorted_point = rotated_point
        x = distorted_point[0] * self.xfocal_length + self.principal_point[0]
        y = distorted_point[1] * self.yfocal_length + self.principal_point[1]
        return np.array([x, y, 1])

    def scale_resolution(self, factor):
        """
        Adapts the internal parameters for image resolution changes
        :param factor: scaling factor
        """
        self.xfocal_length = self.xfocal_length * factor
        self.yfocal_length = self.yfocal_length * factor
        self.image_width = self.image_width * factor
        self.image_height = self.image_height * factor

        self.principal_point = (self.image_width / 2, self.image_height / 2)

        self.calibration = np.array([
            [self.xfocal_length, 0, self.principal_point[0]],
            [0, self.yfocal_length, self.principal_point[1]],
            [0, 0, 1]
        ], dtype='float')

    def draw_corners(self, image, color=(0, 255, 0)):
        """
        Draw the corners of a standard soccer pitch in the image.
        :param image: cv image
        :param color
        :return: the image mat modified.
        """
        field = SoccerField()
        for pt3D in field.point_dict.values():
            projected = self.project_point(pt3D)
            if projected[2] == 0.:
                continue
            projected /= projected[2]
            if 0 < projected[0] < self.image_width and 0 < projected[1] < self.image_height:
                cv.circle(image, (int(projected[0]), int(projected[1])), 3, color, 2)
        return image

    def draw_pitch(self, image, color=(0, 255, 0)):
        """
        Draws all the lines of the pitch on the image.
        :param image
        :param color
        :return: modified image
        """
        field = SoccerField()

        polylines = field.sample_field_points()
        for line in polylines.values():
            prev_point = self.project_point(line[0])
            for point in line[1:]:
                projected = self.project_point(point)
                if projected[2] == 0.:
                    continue
                projected /= projected[2]
                if 0 < projected[0] < self.image_width and 0 < projected[1] < self.image_height:
                    cv.line(image, (int(prev_point[0]), int(prev_point[1])), (int(projected[0]), int(projected[1])),
                            color, 1)
                prev_point = projected
        return image

    def draw_colorful_pitch(self, image, palette):
        """
        Draws all the lines of the pitch on the image, each line color is specified by the palette argument.

        :param image:
        :param palette: dictionary associating line classes names with their BGR color.
        :return: modified image
        """
        field = SoccerField()

        polylines = field.sample_field_points()
        for key, line in polylines.items():
            if key not in palette.keys():
                print(f"Can't draw {key}")
                continue
            prev_point = self.project_point(line[0])
            for point in line[1:]:
                projected = self.project_point(point)
                if projected[2] == 0.:
                    continue
                projected /= projected[2]
                if 0 < projected[0] < self.image_width and 0 < projected[1] < self.image_height:
                    # BGR color
                    cv.line(image, (int(prev_point[0]), int(prev_point[1])), (int(projected[0]), int(projected[1])),
                            palette[key][::-1], 1)
                prev_point = projected
        return image

    def estimate_calibration_matrix_from_plane_homography(self, homography):
        """
        This method initializes the calibration matrix from the homography between the world plane of the pitch
        and the image. It is based on the extraction of the calibration matrix from the homography (Algorithm 8.2 of
        Multiple View Geometry in computer vision, p225). The extraction is sensitive to noise, which is why we keep the
        principal point in the middle of the image rather than using the one extracted by this method.
        :param homography: homography between the world plane of the pitch and the image
        """
        H = np.reshape(homography, (9,))
        A = np.zeros((5, 6))
        A[0, 1] = 1.
        A[1, 0] = 1.
        A[1, 2] = -1.
        A[2, 3] = 9.0 / 16.0
        A[2, 4] = -1.0
        A[3, 0] = H[0] * H[1]
        A[3, 1] = H[0] * H[4] + H[1] * H[3]
        A[3, 2] = H[3] * H[4]
        A[3, 3] = H[0] * H[7] + H[1] * H[6]
        A[3, 4] = H[3] * H[7] + H[4] * H[6]
        A[3, 5] = H[6] * H[7]
        A[4, 0] = H[0] * H[0] - H[1] * H[1]
        A[4, 1] = 2 * H[0] * H[3] - 2 * H[1] * H[4]
        A[4, 2] = H[3] * H[3] - H[4] * H[4]
        A[4, 3] = 2 * H[0] * H[6] - 2 * H[1] * H[7]
        A[4, 4] = 2 * H[3] * H[6] - 2 * H[4] * H[7]
        A[4, 5] = H[6] * H[6] - H[7] * H[7]

        u, s, vh = np.linalg.svd(A)
        w = vh[-1]
        W = np.zeros((3, 3))
        W[0, 0] = w[0] / w[5]
        W[0, 1] = w[1] / w[5]
        W[0, 2] = w[3] / w[5]
        W[1, 0] = w[1] / w[5]
        W[1, 1] = w[2] / w[5]
        W[1, 2] = w[4] / w[5]
        W[2, 0] = w[3] / w[5]
        W[2, 1] = w[4] / w[5]
        W[2, 2] = w[5] / w[5]

        try:
            Ktinv = np.linalg.cholesky(W)
        except np.linalg.LinAlgError:
            K = np.eye(3)
            return False, K

        K = np.linalg.inv(np.transpose(Ktinv))
        K /= K[2, 2]

        self.xfocal_length = K[0, 0]
        self.yfocal_length = K[1, 1]
        # the principal point estimated by this method is very noisy, better keep it in the center of the image
        self.principal_point = (self.image_width / 2, self.image_height / 2)
        # self.principal_point = (K[0,2], K[1,2])
        self.calibration = np.array([
            [self.xfocal_length, 0, self.principal_point[0]],
            [0, self.yfocal_length, self.principal_point[1]],
            [0, 0, 1]
        ], dtype='float')
        return True, K


def normalization_transform(points):
    """
    Computes the similarity transform such that the list of points is centered around (0,0) and that its distance to the
    center is sqrt(2).
    :param points: point cloud that we wish to normalize
    :return: the affine transformation matrix
    """
    center = np.mean(points, axis=0)

    d = 0.
    nelems = 0
    for p in points:
        nelems += 1
        x = p[0] - center[0]
        y = p[1] - center[1]
        di = np.sqrt(x ** 2 + y ** 2)
        d += (di - d) / nelems

    if d <= 0.:
        s = 1.
    else:
        s = np.sqrt(2) / d
    T = np.zeros((3, 3))
    T[0, 0] = s
    T[0, 2] = -s * center[0]
    T[1, 1] = s
    T[1, 2] = -s * center[1]
    T[2, 2] = 1
    return T


def estimate_homography_from_line_correspondences(lines, T1=np.eye(3), T2=np.eye(3)):
    """
    Given lines correspondences, computes the homography that maps best the two set of lines.
    :param lines: list of pair of 2D lines matches.
    :param T1: Similarity transform to normalize the elements of the source reference system
    :param T2: Similarity transform to normalize the elements of the target reference system
    :return: boolean to indicate success or failure of the estimation, homography
    """
    homography = np.eye(3)
    A = np.zeros((len(lines) * 2, 9))

    for i, line_pair in enumerate(lines):
        src_line = np.transpose(np.linalg.inv(T1)) @ line_pair[0]
        target_line = np.transpose(np.linalg.inv(T2)) @ line_pair[1]
        u = src_line[0]
        v = src_line[1]
        w = src_line[2]

        x = target_line[0]
        y = target_line[1]
        z = target_line[2]

        A[2 * i, 0] = 0
        A[2 * i, 1] = x * w
        A[2 * i, 2] = -x * v
        A[2 * i, 3] = 0
        A[2 * i, 4] = y * w
        A[2 * i, 5] = -v * y
        A[2 * i, 6] = 0
        A[2 * i, 7] = z * w
        A[2 * i, 8] = -v * z

        A[2 * i + 1, 0] = x * w
        A[2 * i + 1, 1] = 0
        A[2 * i + 1, 2] = -x * u
        A[2 * i + 1, 3] = y * w
        A[2 * i + 1, 4] = 0
        A[2 * i + 1, 5] = -u * y
        A[2 * i + 1, 6] = z * w
        A[2 * i + 1, 7] = 0
        A[2 * i + 1, 8] = -u * z

    try:
        u, s, vh = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return False, homography
    v = np.eye(3)
    has_positive_singular_value = False
    for i in range(s.shape[0] - 1, -2, -1):
        v = np.reshape(vh[i], (3, 3))

        if s[i] > 0:
            has_positive_singular_value = True
            break

    if not has_positive_singular_value:
        return False, homography

    homography = np.reshape(v, (3, 3))
    homography = np.linalg.inv(T2) @ homography @ T1
    homography /= homography[2, 2]

    return True, homography


def draw_pitch_homography(image, homography):
    """
    Draws points along the soccer pitch markings elements in the image based on the homography projection.
    /!\ This function assumes that the resolution of the image is 540p.
    :param image
    :param homography: homography that captures the relation between the world pitch plane and the image
    :return: modified image
    """
    field = SoccerField()
    polylines = field.sample_field_points()
    for line in polylines.values():

        for point in line:
            if point[2] == 0.:
                hp = np.array((point[0], point[1], 1.))
                projected = homography @ hp
                if projected[2] == 0.:
                    continue
                projected /= projected[2]
                if 0 < projected[0] < 960 and 0 < projected[1] < 540:
                    cv.circle(image, (int(projected[0]), int(projected[1])), 1, (255, 0, 0), 1)

    return image


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Baseline for camera parameters extraction')

    parser.add_argument('-s', '--soccernet', default="/mnt/ahl03/data/football/SoccerNet/data/", type=str,
                        help='Path to the SoccerNet-V3 dataset folder')
    parser.add_argument('-p', '--prediction', default="/mnt/ahl03/users/fmg/results/soccernet_deeplab_baseline",
                        required=False, type=str,
                        help="Path to the prediction folder")
    parser.add_argument('--split', required=False, type=str, default="test", help='Select the split of data')
    parser.add_argument('--resolution_width', required=False, type=int, default=960,
                        help='width resolution of the images')
    parser.add_argument('--resolution_height', required=False, type=int, default=540,
                        help='height resolution of the images')
    args = parser.parse_args()

    list_games = getListGames(args.split, task="frames")
    field = SoccerField()

    with tqdm(enumerate(list_games), total=len(list_games), ncols=160) as t:
        for i, game in t:
            prediction_file = os.path.join(args.prediction, game, "prediction_extremities.json")

            if not os.path.exists(prediction_file):
                continue

            with open(prediction_file, 'r') as f:
                predictions = json.load(f)

            camera_predictions = dict()
            for img in predictions.keys():
                image_path = os.path.join(args.soccernet, game, "v3_frames", img)
                # cv_image = cv.imread(image_path)
                # cv_image = cv.resize(cv_image, (args.resolution_width, args.resolution_height))

                line_matches = []
                potential_3d_2d_matches = {}
                src_pts = []
                success = False
                for k, v in predictions[img].items():
                    if k == 'Circle central' or "unknown" in k:
                        continue
                    P3D1 = field.line_extremities_keys[k][0]
                    P3D2 = field.line_extremities_keys[k][1]
                    p1 = np.array([v[0]['x'] * args.resolution_width, v[0]['y'] * args.resolution_height, 1.])
                    p2 = np.array([v[1]['x'] * args.resolution_width, v[1]['y'] * args.resolution_height, 1.])
                    src_pts.extend([p1, p2])
                    if P3D1 in potential_3d_2d_matches.keys():
                        potential_3d_2d_matches[P3D1].extend([p1, p2])
                    else:
                        potential_3d_2d_matches[P3D1] = [p1, p2]
                    if P3D2 in potential_3d_2d_matches.keys():
                        potential_3d_2d_matches[P3D2].extend([p1, p2])
                    else:
                        potential_3d_2d_matches[P3D2] = [p1, p2]

                    start = (int(p1[0]), int(p1[1]))
                    end = (int(p2[0]), int(p2[1]))
                    # cv.line(cv_image, start, end, (0, 0, 255), 1)

                    line = np.cross(p1, p2)
                    if np.isnan(np.sum(line)) or np.isinf(np.sum(line)):
                        continue
                    line_pitch = field.get_2d_homogeneous_line(k)
                    if line_pitch is not None:
                        line_matches.append((line_pitch, line))

                if len(line_matches) >= 4:
                    target_pts = [field.point_dict[k][:2] for k in potential_3d_2d_matches.keys()]
                    T1 = normalization_transform(target_pts)
                    T2 = normalization_transform(src_pts)
                    success, homography = estimate_homography_from_line_correspondences(line_matches, T1, T2)
                    if success:
                        # cv_image = draw_pitch_homography(cv_image, homography)

                        cam = Camera(args.resolution_width, args.resolution_height)
                        success = cam.from_homography(homography)
                        if success:
                            point_matches = []
                            added_pts = set()
                            for k, potential_matches in potential_3d_2d_matches.items():
                                p3D = field.point_dict[k]
                                projected = cam.project_point(p3D)

                                if 0 < projected[0] < args.resolution_width and 0 < projected[
                                    1] < args.resolution_height:
                                    dist = np.zeros(len(potential_matches))
                                    for i, potential_match in enumerate(potential_matches):
                                        dist[i] = np.sqrt((projected[0] - potential_match[0]) ** 2 + (
                                                projected[1] - potential_match[1]) ** 2)
                                    selected = np.argmin(dist)
                                    if dist[selected] < 100:
                                        point_matches.append((p3D, potential_matches[selected][:2]))

                            if len(point_matches) > 3:
                                cam.refine_camera(point_matches)
                                # cam.draw_colorful_pitch(cv_image, SoccerField.palette)
                                # print(image_path)
                                # cv.imshow("colorful pitch", cv_image)
                                # cv.waitKey(0)

                if success:
                    camera_predictions[img] = cam.to_json_parameters()

            task2_prediction_file = os.path.join(args.prediction, game, "prediction_cameras.json")
            print(task2_prediction_file)
            with open(task2_prediction_file, "w") as f:
                json.dump(camera_predictions, f, indent=4)
