import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class PCCameraNode(Node):
    def __init__(self):
        super().__init__('pc_camera_node')
       	
        self.publisher_ = self.create_publisher(Image, '/camera/color/image_raw', 10)
        self.bridge = CvBridge()
        
        self.cap = cv2.VideoCapture(0,cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.get_logger().error("Cannot open the PC webcam! Check VirtualBox USB settings.")
         
        timer_period = 0.033 # 30 fps circa 
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info("Camera Node is publishing video frames...")

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self.publisher_.publish(msg)
        else:
            self.get_logger().warning("Failed to capture image frame")

def main(args=None):
    rclpy.init(args=args)
    node = PCCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Camera Node.")
    finally:
        node.cap.release()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
