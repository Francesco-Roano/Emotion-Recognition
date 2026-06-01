import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import pyaudio
import numpy as np

class PCMicrophoneNode(Node):
    def __init__(self):
        super().__init__('pc_microphone_node')
       
        self.publisher_ = self.create_publisher(Float32MultiArray, '/microphone/audio_data', 10)
        
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 16000     
        self.CHUNK = 1024     

        self.audio = pyaudio.PyAudio()
        
        try:
            self.stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                input_device_index=17,
                frames_per_buffer=self.CHUNK,
                stream_callback=self.audio_callback
            )
            self.stream.start_stream()
            self.get_logger().info(f"Microphone Node publishing audio at {self.RATE}Hz...")
        except Exception as e:
            self.get_logger().error(f"Failed to open microphone: {e}.")

    def audio_callback(self, in_data, frame_count, time_info, status):
        try:
            audio_np = np.frombuffer(in_data, dtype=np.float32)
            if self.CHANNELS==2:
                audio_np = audio_np[::2] # just take left audio signal
            
            msg = Float32MultiArray()
            msg.data = audio_np.tolist()
            self.publisher_.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Callback error: {e}")
        return (None, pyaudio.paContinue)
        

def main(args=None):
    rclpy.init(args=args)
    node = PCMicrophoneNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Microphone Node.")
    finally:
        if hasattr(node, 'stream'):
            node.stream.stop_stream()
            node.stream.close()
        node.audio.terminate()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
