from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.geometry.transform import translate_longitudinally_and_laterally


def get_front_left_corner(center_pose: StateSE2, half_length: float, half_width: float) -> Point2D:
    """
    根据车辆中心姿态和尺寸计算左前角点位置。
    :param center_pose: SE2 姿态表示的车辆中心。
    :param half_length: [m] 车辆轮廓的一半长度。
    :param half_width: [m] 车辆轮廓的一半宽度。
    :return: 平移后的坐标（Point2D）。
    """
    return translate_longitudinally_and_laterally(center_pose, half_length, half_width).point


def get_front_right_corner(center_pose: StateSE2, half_length: float, half_width: float) -> Point2D:
    """
    根据车辆中心姿态和尺寸计算右前角点位置。
    :param center_pose: SE2 姿态表示的车辆中心。
    :param half_length: [m] 车辆轮廓的一半长度。
    :param half_width: [m] 车辆轮廓的一半宽度。
    :return: 平移后的坐标（Point2D）。
    """
    return translate_longitudinally_and_laterally(center_pose, half_length, -half_width).point


def get_rear_left_corner(center_pose: StateSE2, half_length: float, half_width: float) -> Point2D:
    """
    根据车辆中心姿态和尺寸计算左后角点位置。
    :param center_pose: SE2 姿态表示的车辆中心。
    :param half_length: [m] 车辆轮廓的一半长度。
    :param half_width: [m] 车辆轮廓的一半宽度。
    :return: 平移后的坐标（Point2D）。
    """
    return translate_longitudinally_and_laterally(center_pose, -half_length, half_width).point


def get_rear_right_corner(center_pose: StateSE2, half_length: float, half_width: float) -> Point2D:
    """
    根据车辆中心姿态和尺寸计算右后角点位置。
    :param center_pose: SE2 姿态表示的车辆中心。
    :param half_length: [m] 车辆轮廓的一半长度。
    :param half_width: [m] 车辆轮廓的一半宽度。
    :return: 平移后的坐标（Point2D）。
    """
    return translate_longitudinally_and_laterally(center_pose, -half_length, -half_width).point