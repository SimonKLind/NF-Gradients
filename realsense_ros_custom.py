#!/usr/bin/env python

import rospy
import pyrealsense2 as rs
import numpy as np
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from vision_msgs.srv import RealsenseGetParams, RealsenseGetParamsResponse, GetIntrinsicParams, GetIntrinsicParamsResponse
from vision_msgs.msg import RealsenseParamInfo, RealsenseSetParams, IntrinsicParams
from std_msgs.msg import Empty, Bool
import threading

class RealsenseParam:
    def __init__(self, name, opt_enum, value=0, default=0, step=0, vmin=0, vmax=0):
        self.name = name
        self.opt_enum = opt_enum
        self.default = default
        self.step = step
        self.min = vmin
        self.max = vmax

class RealsenseParams:
    def __init__(self, sensor, model):
        self.sensor = sensor

        if model == 'D515':
            # Realsense D515 settings
            params = [
                # We just initialize all values to zero, they are filled in below
                RealsenseParam('backlight_compensation', rs.option.backlight_compensation),
                RealsenseParam('brightness',             rs.option.brightness            ),
                RealsenseParam('contrast',               rs.option.contrast              ),
                RealsenseParam('exposure',               rs.option.exposure              ),
                RealsenseParam('gain',                   rs.option.gain                  ),
                RealsenseParam('saturation',             rs.option.saturation            ),
                RealsenseParam('sharpness',              rs.option.sharpness             ),
                RealsenseParam('white_balance',          rs.option.white_balance         )
            ]
            #self.lo = np.array([0.0, -64.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2800.0], dtype=np.float32)
            #self.hi = np.array([255.0, 64.0, 100.0, 666.0, 4096.0, 100.0, 100.0, 6500.0], dtype=np.float32)
        elif model == 'D435':
            # Realsense D435 settings
            params = [
                # We just initialize all values to zero, they are filled in below
                RealsenseParam('backlight_compensation', rs.option.backlight_compensation),
                RealsenseParam('brightness',             rs.option.brightness            ),
                RealsenseParam('contrast',               rs.option.contrast              ),
                RealsenseParam('exposure',               rs.option.exposure              ),
                RealsenseParam('gain',                   rs.option.gain                  ),
                RealsenseParam('gamma',                  rs.option.gamma                 ),
                RealsenseParam('saturation',             rs.option.saturation            ),
                RealsenseParam('sharpness',              rs.option.sharpness             ),
                RealsenseParam('white_balance',          rs.option.white_balance         ),
            ]
            #self.lo = np.array([0.0, -64.0, 0.0, 1.0, 0.0, 100.0, 0.0, 0.0, 2800.0], dtype=np.float32)
            #self.hi = np.array([1.0, 64.0, 100.0, 666.0, 128.0, 500.0, 100.0, 100.0, 6500.0], dtype=np.float32)
        else:
            raise Exception('Unrecognized camera model')

        self.params = { p.name: p for p in params }

        self.n = len(self.params)
        for p in self.params.values():
            r = self.sensor.get_option_range(p.opt_enum)
            p.value = self.sensor.get_option(p.opt_enum)
            p.default = r.default
            p.step = r.step
            p.min = r.min
            p.max = r.max

    def update(self):
        for p in self.params.values():
            self.sensor.set_option(p.opt_enum, p.value)

    def reset(self):
        for p in self.params.values():
            p.value = p.default
            self.sensor.set_option(p.opt_enum, p.value)

    def random(self):
        for p in self.params.values():
            nstep = (p.max - p.min) / p.step
            r = np.random.randint(nstep)
            p.value = (p.min + r * p.step)
            self.sensor.set_option(p.opt_enum, p.value)

    def set(self, xs):
        for name in xs.keys() & self.params.keys():
            p = self.params[name]
            p.value = xs[name]
            self.sensor.set_option(p.opt_enum, p.value)

    def set_params_callback(self, data):
        for x in data.params:
            name, value = x.name, x.value # Manual destructuring because ros doesn't make messages iterable by default...

            if name not in self.params: continue

            p = self.params[name]

            # Round to nearest whole steps
            nsteps = round((value - p.min) / p.step)
            value = p.min + nsteps * p.step
            
            # Clamp to [min, max]
            p.value = max(p.min, min(value, p.max))

            self.sensor.set_option(p.opt_enum, p.value)

    def reset_params_callback(self, data):
        for p in self.params.values():
            p.value = p.default
            self.sensor.set_option(p.opt_enum, p.value)

    def auto_exposure_callback(self, data):
        self.sensor.set_option(rs.option.enable_auto_exposure, float(data.data))

    def auto_white_balance_callback(self, data):
        self.sensor.set_option(rs.option.enable_auto_white_balance, float(data.data))

    def get_params_callback(self, data):
        ret = RealsenseGetParamsResponse()
        ret.params = [RealsenseParamInfo(p.name, p.value, p.default, p.min, p.max) for p in self.params.values()]
        return ret

def has_rgb(device):
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            return True
    return False

class RealsenseCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()

        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()

        if not has_rgb(device):
            raise Exception('Failed to find RGB camera')

        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        profile = self.pipeline.start(config)

        self.rgb_sensor = profile.get_device().first_color_sensor()

        self.depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()

        self.align = rs.align(rs.stream.color)

        for s in profile.get_streams():
            if s.stream_name() == 'Color':
                v = s.as_video_stream_profile()
                i = v.get_intrinsics()
                self.fx = i.fx
                self.fy = i.fy
                self.cx = i.ppx
                self.cy = i.ppy
                break

    def get_images(self):
        depth_frame, rgb_frame = None, None
        while depth_frame is None or rgb_frame is None:
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            depth_frame = aligned.get_depth_frame()
            rgb_frame = aligned.get_color_frame()

        depth = np.asanyarray(depth_frame.get_data())
        depth = (depth * 1000.0 * self.depth_scale).astype(np.uint16)

        rgb = np.asanyarray(rgb_frame.get_data())

        return rgb, depth

    def discard(self, n):
        for _ in range(n): self.pipeline.wait_for_frames()

    def intrinsic_params_callback(self, data):
        return GetIntrinsicParamsResponse(IntrinsicParams(self.fx, self.fy, self.cx, self.cy))
        
class BoolWrapper:
    def __init__(self, val):
        self.value = val

class ImagePublisher(threading.Thread):
    def __init__(self, camera, running):
        super().__init__()
        self.camera = camera
        self.running = running

        self.rgb_pub = rospy.Publisher('/realsense/rgb/image_raw', Image, queue_size=1)
        self.depth_pub = rospy.Publisher('/realsense/aligned_depth_to_color/image_raw', Image, queue_size=1)
        self.bridge = CvBridge()

    def run(self):
        while self.running.value:
            rgb, depth = self.camera.get_images()
            try:
                rgb_msg = self.bridge.cv2_to_imgmsg(rgb, 'passthrough')
                depth_msg = self.bridge.cv2_to_imgmsg(depth, 'passthrough')
                self.rgb_pub.publish(rgb_msg)
                self.depth_pub.publish(depth_msg)
            except Exception as e:
                print(e)

def main():
    rospy.init_node('realsense_ros_wrapper')

    camera = RealsenseCamera()
    params = RealsenseParams(camera.rgb_sensor, 'D435')
    running = BoolWrapper(True)

    image_pub = ImagePublisher(camera, running)
    image_pub.start()

    rospy.Service('/realsense/rgb/get_params', RealsenseGetParams, params.get_params_callback)
    rospy.Service('/realsense/rgb/get_intrinsic_params', GetIntrinsicParams, camera.intrinsic_params_callback)

    rospy.Subscriber('/realsense/rgb/set_params', RealsenseSetParams, params.set_params_callback)
    rospy.Subscriber('/realsense/rgb/reset_params', Empty, params.reset_params_callback)
    rospy.Subscriber('/realsense/rgb/auto_exposure', Bool, params.auto_exposure_callback)
    rospy.Subscriber('/realsense/rgb/auto_white_balance', Bool, params.auto_white_balance_callback)

    rospy.spin()

    running.value = False
    image_pub.join()

if __name__ == '__main__': main()
