import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs_py import point_cloud2
import open3d as o3d
import numpy as np


class ClusterNode(Node):
    def __init__(self):
        super().__init__('cluster_points_node')

        # Subscribe to point cloud topic
        self.subscription = self.create_subscription(
            PointCloud2,
            '/robot/rgb_camera/points',
            self.pointcloud_callback,
            10)

        # Publisher for bounding boxes / obstacles
        self.marker_pub = self.create_publisher(MarkerArray, '/clusters', 10)

        self.get_logger().info('ClusterNode started. Listening on /robot/depth_camera/points')

    def ros_to_numpy(self, cloud_msg):
        points = []
        for p in point_cloud2.read_points(cloud_msg, field_names=('x', 'y', 'z'), skip_nans=True):
            if np.isfinite(p[0]) and np.isfinite(p[1]) and np.isfinite(p[2]):
                points.append([p[0], p[1], p[2]])
        return np.array(points, dtype=np.float64)

    def pointcloud_callback(self, msg):
        np_points = self.ros_to_numpy(msg)
        num_points = np_points.shape[0]

        if num_points == 0:
            self.get_logger().warn('Received empty or invalid point cloud')
            return

        self.get_logger().info(f"Received {num_points} valid points")

        # Convert to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np_points)

        # Downsample
        try:
            pcd = pcd.voxel_down_sample(voxel_size=0.05)
        except RuntimeError as e:
            self.get_logger().warn(f"Downsampling skipped: {str(e)}")
            return

        # Cluster points (DBSCAN)
        if len(pcd.points) == 0:
            self.get_logger().warn("No points left after downsampling")
            return

        labels = np.array(pcd.cluster_dbscan(eps=0.3, min_points=30, print_progress=False))
        num_clusters = int(labels.max()) + 1 if labels.size > 0 else 0
        self.get_logger().info(f'Found {num_clusters} clusters')

        if num_clusters == 0:
            return

        markers = MarkerArray()
        for cluster_id in range(num_clusters):
            cluster_points = np.asarray(pcd.points)[labels == cluster_id]
            if cluster_points.size == 0:
                continue

            xmin, ymin, zmin = cluster_points.min(axis=0)
            xmax, ymax, zmax = cluster_points.max(axis=0)

            marker = Marker()
            marker.header.frame_id = msg.header.frame_id
            marker.id = cluster_id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = (xmin + xmax) / 2.0
            marker.pose.position.y = (ymin + ymax) / 2.0
            marker.pose.position.z = (zmin + zmax) / 2.0
            marker.scale.x = xmax - xmin
            marker.scale.y = ymax - ymin
            marker.scale.z = zmax - zmin
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.4

            markers.markers.append(marker)

        self.marker_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = ClusterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
