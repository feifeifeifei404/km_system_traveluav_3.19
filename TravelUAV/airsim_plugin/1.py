import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

class CloudChecker(Node):
    def __init__(self):
        super().__init__('cloud_checker')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/cloud_registered',
            self.cloud_callback,
            10)
        self.received = False

    def cloud_callback(self, msg):
        points = list(pc2.read_points(msg, field_names=("x","y","z"), skip_nans=True))
        print(f"Received point cloud with {len(points)} points")
        self.received = True
        # 自动退出
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = CloudChecker()
    while not node.received:
        rclpy.spin_once(node, timeout_sec=1)
    node.destroy_node()

if __name__ == "__main__":
    main()