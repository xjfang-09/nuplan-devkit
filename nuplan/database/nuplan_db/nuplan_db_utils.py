from dataclasses import dataclass


@dataclass(frozen=True)
class SensorDataSource:
    """
    包含查询数据库文件以提取传感器数据的参数的类。

    例如，对于查询激光雷达数据，属性将是：
    table: lidar_pc
    sensor_table: lidar
    sensor_token_column: lidar_token (这是存储传感器令牌的 `table` 中的列名)
    channel: MergedPointCloud
    """

    table: str
    sensor_table: str
    sensor_token_column: str
    channel: str

    def __post_init__(self) -> None:
        """检查提供的表是否兼容"""
        if self.table == 'lidar_pc':
            assert (
                self.sensor_table == "lidar"
            ), f"与表 {self.table} 不兼容的 sensor_table: {self.sensor_table}"
        elif self.table == 'image':
            assert (
                self.sensor_table == "camera"
            ), f"与表 {self.table} 不兼容的 sensor_table: {self.sensor_table}"
        else:
            raise ValueError(f"未知的传感器表请求: {self.table}!")

        assert (
            self.sensor_token_column == f"{self.sensor_table}_token"
        ), f"与 sensor_table {self.sensor_table} 不兼容的 sensor_token_column: {self.sensor_token_column}"


def get_lidarpc_sensor_data() -> SensorDataSource:
    """
    构建用于 lidar_pc 的 SensorDataSource。
    :return: lidar_pc 的查询参数。
    """
    return SensorDataSource('lidar_pc', 'lidar', 'lidar_token', 'MergedPointCloud')


def get_camera_channel_sensor_data(channel: str) -> SensorDataSource:
    """
    构建来自指定通道的图像的 SensorDataSource。
    :param channel: 要选择的通道。
    :return: 图像的查询参数。
    """
    return SensorDataSource('image', 'camera', 'camera_token', channel)