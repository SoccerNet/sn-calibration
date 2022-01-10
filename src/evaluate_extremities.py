import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from SoccerNet.utils import getListGames
from tqdm import tqdm

from src.soccerpitch import SoccerPitch


def distance(point1, point2):
    """
    Computes euclidian distance between 2D points
    :param point1
    :param point2
    :return: euclidian distance between point1 and point2
    """
    diff = np.array([point1['x'], point1['y']]) - np.array([point2['x'], point2['y']])
    sq_dist = np.square(diff)
    return np.sqrt(sq_dist.sum())


def mirror_labels(lines_dict):
    """
    Replace each line class key of the dictionary with its opposite element according to a central projection by the
    soccer pitch center
    :param lines_dict: dictionary whose keys will be mirrored
    :return: Dictionary with mirrored keys and same values
    """
    mirrored_dict = dict()
    for line_class, value in lines_dict.items():
        mirrored_dict[SoccerPitch.symetric_classes[line_class]] = value
    return mirrored_dict


def evaluate_detection_prediction(groundtruth_lines, detected_lines, threshold=2.):
    """
    Evaluates the prediction of extremities. The extremities associated to a class are unordered. The extremities of the
    "Circle central" element is not well-defined for this task, thus this class is ignored.
    Computes confusion matrices for a level of precision specified by the threshold.
    A groundtruth extremity point is correctly classified if it lies at less than threshold pixels from the
    corresponding extremity point of the prediction of the same class.
    Computes also the euclidian distance between each predicted extremity and its closest groundtruth extremity, when
    both the groundtruth and the prediction contain the element class.

    :param projected_lines: dictionary of detected lines classes as keys and associated predicted extremities as values
    :param groundtruth_lines: dictionary of annotated lines classes as keys and associated annotated points as values
    :param threshold: distance in pixels that distinguishes good matches from bad ones
    :return: confusion matrix, per class confusion matrix & per class localization errors
    """
    confusion_mat = np.zeros((2, 2), dtype=np.float32)
    per_class_confusion = {}
    errors_dict = {}
    detected_classes = set(detected_lines.keys())
    groundtruth_classes = set(groundtruth_lines.keys())

    false_positives_classes = detected_classes - groundtruth_classes
    for false_positive_class in false_positives_classes:
        false_positives = len(detected_lines[false_positive_class])
        confusion_mat[0, 1] += false_positives
        per_class_confusion[false_positive_class] = np.array([[0., false_positives], [0., 0.]])

    false_negatives_classes = groundtruth_classes - detected_classes
    for false_negatives_class in false_negatives_classes:
        false_negatives = len(groundtruth_lines[false_negatives_class])
        confusion_mat[1, 0] += false_negatives
        per_class_confusion[false_negatives_class] = np.array([[0., 0.], [false_negatives, 0.]])

    common_classes = detected_classes - false_positives_classes

    for detected_class in common_classes:
        if detected_class == "Circle center":
            continue
        detected_points = detected_lines[detected_class]

        groundtruth_points = groundtruth_lines[detected_class]

        groundtruth_extremities = [groundtruth_points[0], groundtruth_points[-1]]
        predicted_extremities = [detected_points[0], detected_points[-1]]
        per_class_confusion[detected_class] = np.zeros((2, 2))

        dist1 = distance(groundtruth_extremities[0], predicted_extremities[0])
        dist1rev = distance(groundtruth_extremities[1], predicted_extremities[0])

        dist2 = distance(groundtruth_extremities[1], predicted_extremities[1])
        dist2rev = distance(groundtruth_extremities[0], predicted_extremities[1])
        if dist1rev <= dist1 and dist2rev <= dist2:
            # reverse order
            dist1 = dist1rev
            dist2 = dist2rev

        errors_dict[detected_class] = [dist1, dist2]

        if dist1 < threshold:
            confusion_mat[0, 0] += 1
            per_class_confusion[detected_class][0, 0] += 1
        else:
            # treat too far detections as false positives
            confusion_mat[0, 1] += 1
            per_class_confusion[detected_class][0, 1] += 1

        if dist2 < threshold:
            confusion_mat[0, 0] += 1
            per_class_confusion[detected_class][0, 0] += 1

        else:
            # treat too far detections as false positives
            confusion_mat[0, 1] += 1
            per_class_confusion[detected_class][0, 1] += 1

    return confusion_mat, per_class_confusion, errors_dict


def scale_points(points_dict, s_width, s_height):
    """
    Scale points by s_width and s_height factors
    :param points_dict: dictionary of annotations/predictions with normalized point values
    :param s_width: width scaling factor
    :param s_height: height scaling factor
    :return: dictionary with scaled points
    """
    scaled_dict = {}
    for img, lines in points_dict.items():
        line_dict = {}
        for line_class, points in lines.items():
            scaled_points = []
            for point in points:
                new_point = {'x': point['x'] * s_width, 'y': point['y'] * s_height}
                scaled_points.append(new_point)
            if len(scaled_points):
                line_dict[line_class] = scaled_points
        if len(line_dict):
            scaled_dict[img] = line_dict
    return scaled_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test')

    parser.add_argument('-s', '--soccernet', default="./annotations", type=str,
                        help='Path to the SoccerNet-V3 dataset folder')
    parser.add_argument('-p', '--prediction', default="/mnt/ahl03/users/fmg/results/soccernet_deeplab_baseline",
                        required=False, type=str,
                        help="Path to the prediction folder")
    parser.add_argument('-t', '--threshold', default=10, required=False, type=int,
                        help="Accuracy threshold in pixels")
    parser.add_argument('--split', required=False, type=str, default="test", help='Select the split of data')
    parser.add_argument('--resolution_width', required=False, type=int, default=960,
                        help='width resolution of the images')
    parser.add_argument('--resolution_height', required=False, type=int, default=540,
                        help='height resolution of the images')
    args = parser.parse_args()

    accuracies = []
    precisions = []
    recalls = []
    dict_errors = {}
    per_class_confusion_dict = {}

    list_games = getListGames(args.split, task="frames")

    with tqdm(enumerate(list_games), total=len(list_games), ncols=160) as t:
        for i, game in t:
            annotation_file = os.path.join(args.soccernet, game, "labels-calibration.json")
            prediction_file = os.path.join(args.prediction, game, "prediction_extremities.json")

            if not os.path.exists(prediction_file):
                continue

            with open(annotation_file, 'r') as f:
                line_annotations = json.load(f)

            with open(prediction_file, 'r') as f:
                predictions = json.load(f)

            predictions = scale_points(predictions, args.resolution_width, args.resolution_height)
            line_annotations = scale_points(line_annotations, args.resolution_width, args.resolution_height)

            for img in line_annotations.keys():

                if img not in predictions.keys():
                    accuracies.append(0.)
                    continue
                img_prediction = predictions[img]
                img_groundtruth = line_annotations[img]
                confusion1, per_class_conf1, reproj_errors1 = evaluate_detection_prediction(img_prediction,
                                                                                            img_groundtruth,
                                                                                            args.threshold)
                confusion2, per_class_conf2, reproj_errors2 = evaluate_detection_prediction(img_prediction,
                                                                                            mirror_labels(
                                                                                                img_groundtruth),
                                                                                            args.threshold)

                accuracy1, accuracy2 = 0., 0.
                if confusion1.sum() > 0:
                    accuracy1 = confusion1[0, 0] / confusion1.sum()

                if confusion2.sum() > 0:
                    accuracy2 = confusion2[0, 0] / confusion2.sum()

                if accuracy1 > accuracy2:
                    accuracy = accuracy1
                    confusion = confusion1
                    per_class_conf = per_class_conf1
                    reproj_errors = reproj_errors1
                else:
                    accuracy = accuracy2
                    confusion = confusion2
                    per_class_conf = per_class_conf2
                    reproj_errors = reproj_errors2

                accuracies.append(accuracy)
                if confusion[0, :].sum() > 0:
                    precision = confusion[0, 0] / (confusion[0, :].sum())
                    precisions.append(precision)
                if (confusion[0, 0] + confusion[1, 0]) > 0:
                    recall = confusion[0, 0] / (confusion[0, 0] + confusion[1, 0])
                    recalls.append(recall)

                for line_class, errors in reproj_errors.items():
                    if line_class in dict_errors.keys():
                        dict_errors[line_class].extend(errors)
                    else:
                        dict_errors[line_class] = errors

                for line_class, confusion_mat in per_class_conf.items():
                    if line_class in per_class_confusion_dict.keys():
                        per_class_confusion_dict[line_class] += confusion_mat
                    else:
                        per_class_confusion_dict[line_class] = confusion_mat

    mRecall = np.mean(recalls)
    sRecall = np.std(recalls)
    medianRecall = np.median(recalls)
    print(
        f" On SoccerNet {args.split} set, recall mean value : {mRecall * 100:2.2f}% with standard deviation of {sRecall * 100:2.2f}% and median of {medianRecall * 100:2.2f}%")

    mPrecision = np.mean(precisions)
    sPrecision = np.std(precisions)
    medianPrecision = np.median(precisions)
    print(
        f" On SoccerNet {args.split} set, precision mean value : {mPrecision * 100:2.2f}% with standard deviation of {sPrecision * 100:2.2f}% and median of {medianPrecision * 100:2.2f}%")

    mAccuracy = np.mean(accuracies)
    sAccuracy = np.std(accuracies)
    medianAccuracy = np.median(accuracies)
    print(
        f" On SoccerNet {args.split} set, accuracy mean value : {mAccuracy * 100:2.2f}% with standard deviation of {sAccuracy * 100:2.2f}% and median of {medianAccuracy * 100:2.2f}%")

    for line_class, confusion_mat in per_class_confusion_dict.items():
        class_accuracy = confusion_mat[0, 0] / confusion_mat.sum()
        class_recall = confusion_mat[0, 0] / (confusion_mat[0, 0] + confusion_mat[1, 0])
        class_precision = confusion_mat[0, 0] / (confusion_mat[0, 0] + confusion_mat[0, 1])
        print(
            f"For class {line_class}, accuracy of {class_accuracy * 100:2.2f}%, precision of {class_precision * 100:2.2f}%  and recall of {class_recall * 100:2.2f}%")

    for k, v in dict_errors.items():
        fig, ax1 = plt.subplots(figsize=(11, 8))
        ax1.hist(v, bins=30, range=(0, 60))
        ax1.set_title(k)
        ax1.set_xlabel("Errors in pixel")
        plt.savefig(f"./results/{k}_detection_error.png")
        plt.close(fig)
