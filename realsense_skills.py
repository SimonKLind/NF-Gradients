from skiros2_skill.core.skill import SkillDescription, SkillBase, Sequential, ParamOptions, Loop
from skiros2_common.core.params import ParamTypes
from skiros2_common.core.primitive import PrimitiveBase
from skiros2_common.core.world_element import Element
import skiros2_common.tools.logger as log
import moveit_commander
import sys
import threading
import math

import rospy
from cv_bridge import CvBridge, CvBridgeError
import tf
import cv2 as cv
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Empty, Bool
from wsg_50_common.srv import Move
from vision_msgs.msg import RealsenseSetParams, RealsenseParam
from vision_msgs.srv import GetIntrinsicParams, YOLODetections, YOLODetectionsRequest, RealsenseGetParams, NVPConfidence, NVPConfidenceRequest, RealsenseGetParams
import numpy as np
import datetime
import os

class RealsenseResetParams(SkillDescription):

    def createDescription(self):
        pass

class realsense_reset_params(PrimitiveBase):

    def createDescription(self):
        self.setDescription(RealsenseResetParams(), self.__class__.__name__)

    def modifyDescription(self, skill):
        pass

    def onPreempt(self):
        return self.fail('Canceled', -1)

    def onStart(self):
        self.pub = rospy.Publisher('/realsense/rgb/reset_params', Empty, queue_size=1)
        self.rate = rospy.Rate(10)
        return True

    def execute(self):
        while self.pub.get_num_connections() <= 0: self.rate.sleep()
        self.pub.publish()
        return self.success('Done')

class RealsenseRandomizeParams(SkillDescription):
    def createDescription(self):
        pass

def random_params(param_specs):
    out = []
    for name, spec in param_specs.items():
        value = spec['min'] + (spec['max']-spec['min']) * np.random.random()
        out.append(RealsenseParam(name, value))
    return out

class realsense_randomize_params(PrimitiveBase):
    def createDescription(self):
        self.setDescription(RealsenseRandomizeParams(), self.__class__.__name__)

    def modifyDescription(self, skill):
        pass

    def onPreempt(self):
        return self.fail('Canceled', -1)

    def onStart(self):
        self.pub = rospy.Publisher('/realsense/rgb/set_params', RealsenseSetParams, queue_size=1)
        self.rate = rospy.Rate(10)
        return True

    def execute(self):
        rospy.wait_for_service('/realsense/rgb/get_params')
        get_params_srv = rospy.ServiceProxy('/realsense/rgb/get_params', RealsenseGetParams)
        res = get_params_srv()
        params = {}
        for p in res.params:
            params[p.name] = {
                'value': p.value,
                'default': p.default,
                'min': p.min,
                'max': p.max
            }

        msg = RealsenseSetParams(random_params(params))
        while self.pub.get_num_connections() <= 0: self.rate.sleep()
        self.pub.publish(msg)

        return self.success('Done')

class RealsenseAutoExposure(SkillDescription):

    def createDescription(self):
        self.addParam("Enabled", True, ParamTypes.Required)

class realsense_auto_exposure(PrimitiveBase):

    def createDescription(self):
        self.setDescription(RealsenseAutoExposure(), self.__class__.__name__)

    def modifyDescription(self, skill):
        pass

    def onPreempt(self):
        return self.fail('Canceled', -1)

    def onStart(self):
        self.pub = rospy.Publisher('/realsense/rgb/auto_exposure', Bool, queue_size=1)
        self.rate = rospy.Rate(10)
        return True

    def execute(self):
        while self.pub.get_num_connections() <= 0: self.rate.sleep()
        self.pub.publish(self.params["Enabled"].value)
        return self.success('Done')

class RealsenseAutoWhiteBalance(SkillDescription):

    def createDescription(self):
        self.addParam("Enabled", True, ParamTypes.Required)

class realsense_auto_white_balance(PrimitiveBase):

    def createDescription(self):
        self.setDescription(RealsenseAutoWhiteBalance(), self.__class__.__name__)

    def modifyDescription(self, skill):
        pass

    def onPreempt(self):
        return self.fail('Canceled', -1)

    def onStart(self):
        self.pub = rospy.Publisher('/realsense/rgb/auto_white_balance', Bool, queue_size=1)
        self.rate = rospy.Rate(10)
        return True

    def execute(self):
        while self.pub.get_num_connections() <= 0: self.rate.sleep()
        self.pub.publish(self.params["Enabled"].value)
        return self.success('Done')

class RealsenseDiscardFrames(SkillDescription):

    def createDescription(self):
        self.addParam("Number", 1, ParamTypes.Required)

class realsense_discard_frames(PrimitiveBase):

    def createDescription(self):
        self.setDescription(RealsenseDiscardFrames(), self.__class__.__name__)

    def modifyDescription(self, skill):
        pass

    def onPreempt(self):
        return self.fail('Canceled', -1)

    def image_callback(self, data):
        self.count += 1

    def onStart(self):
        self.count = 0
        self.sub = rospy.Subscriber('/realsense/rgb/image_raw', Image, self.image_callback)
        return True

    def execute(self):
        if self.count < self.params['Number'].value:
            return self.step('Running...')

        self.sub.unregister()
        return self.success('Done')

class RealsenseSaveRGB(SkillDescription):

    def createDescription(self):
        self.addParam("OutputDir", '/tmp', ParamTypes.Required)

class realsense_save_rgb(PrimitiveBase):

    def createDescription(self):
        self.setDescription(RealsenseSaveRGB(), self.__class__.__name__)

    def modifyDescription(self, skill):
        pass

    def onPreempt(self):
        return self.fail('Canceled', -1)

    def image_callback(self, data):
        if self.block: return
        self.block = True
        try:
            image = self.bridge.imgmsg_to_cv2(data, 'passthrough')
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')
            if not os.path.exists(self.out_dir): os.mkdir(self.out_dir)
            path = os.path.join(self.out_dir, timestamp + '.jpg')
            cv.imwrite(path, image)
            self.saved = True
        except Exception as e:
            self.block = False
            print(e)

    def onStart(self):
        self.bridge = CvBridge()
        self.out_dir = self.params['OutputDir'].value
        self.saved = False
        self.block = False
        self.sub = rospy.Subscriber('/realsense/rgb/image_raw', Image, self.image_callback)
        return True

    def execute(self):
        if not self.saved:
            return self.step('Running...')

        self.sub.unregister()
        return self.success('Done')

class RGBListener:
    def __init__(self, topic='/realsense/rgb/image_raw'):
        self.rate = rospy.Rate(10)
        self.imgmsg = 0.0

        rospy.Subscriber(topic, Image, callback=self.image_callback)

    def image_callback(self, data):
        if self.imgmsg is not None: return
        self.imgmsg = data

    def get(self):
        self.imgmsg = None
        while self.imgmsg is None:
            self.rate.sleep()
        return self.imgmsg


class NVPOptimize(SkillDescription):

    def createDescription(self):
        self.addParam("Population", 50, ParamTypes.Required)
        self.addParam("MutationRate", 0.2, ParamTypes.Required)
        self.addParam("NumIterations", 200, ParamTypes.Required)

        self.addParam("Verbose", False, ParamTypes.Optional)
        self.addParam("CustomParamRange", True, ParamTypes.Optional)
        self.addParam("UseGradImage", False, ParamTypes.Optional)

class nvp_optimize(PrimitiveBase):

    def createDescription(self):
        self.setDescription(NVPOptimize(), self.__class__.__name__)

    def modifyDescription(self, skill):
        pass

    # This is the function we optimize over
    def opt_f(self, values):
        msg = RealsenseSetParams([RealsenseParam(self.names[i], x) for i, x in enumerate(values)])
        self.param_pub.publish(msg)

        # Discarding 3 images seems to be the magic number to
        # ensure all parameters are fully updated
        for _ in range(3):
            self.rgb_listener.get()

        imgmsg = self.rgb_listener.get()
        response = self.conf_srv(imgmsg)

        if not self.use_grad_image:
            return response.confidence

        try:
            grad_image = self.bridge.imgmsg_to_cv2(response.grad_image, 'passthrough')
        except Exception as e:
            grad_image = np.inf
            print(e)

        detections = self.det_srv(imgmsg).detections
        mask = np.zeros_like(grad_image, dtype=bool)
        for d in detections:
            if d.class_name not in self.labels_of_interest: continue
            cx = d.center_x * imgmsg.width
            cy = d.center_y * imgmsg.height
            w = 0.5 * d.width * imgmsg.width
            h = 0.5 * d.height * imgmsg.height
            x = int(cx - w)
            y = int(cy - h)
            x2 = int(cx + w)
            y2 = int(cy + h)
            mask[y:y2, x:x2] = True

        vals = grad_image[mask].flatten()
        if vals.size == 0: vals = grad_image.flatten()

        return -np.mean(vals)

    def run(self):
        verbose = self.params['Verbose'].value
        custom_range = self.params['CustomParamRange'].value
        self.use_grad_image = self.params['UseGradImage'].value

        self.labels_of_interest = {'apple', 'banana', 'scissors', 'fork', 'spoon', 'cup', 'mouse', 'book'}

        self.rgb_listener = RGBListener()

        rospy.wait_for_service('/nvp/get_confidence')
        self.conf_srv = rospy.ServiceProxy('/nvp/get_confidence', NVPConfidence)

        rospy.wait_for_service('/yolo/get_detections')
        self.det_srv = rospy.ServiceProxy('/yolo/get_detections', YOLODetections)

        self.param_pub = rospy.Publisher('/realsense/rgb/set_params', RealsenseSetParams, queue_size=1)

        population_size = self.params['Population'].value
        mutation_rate = self.params['MutationRate'].value
        n_iterations = self.params['NumIterations'].value

        if verbose:
            print('Entering nvp_optimize with params')
            print('Population size:', population_size)
            print('Mutation rate:', mutation_rate)
            print('Number of iterations:', n_iterations)
            print('Grad Image:', self.use_grad_image)

        rospy.wait_for_service('/realsense/rgb/get_params')
        param_srv = rospy.ServiceProxy('/realsense/rgb/get_params', RealsenseGetParams)
        param_specs = param_srv()

        n_vars = len(param_specs.params)

        self.names = [p.name for p in param_specs.params]
        
        default = np.array([p.default for p in param_specs.params])

        if custom_range:
            lo = np.array([0.0, -64.0, 0.0, 1.0, 0.0, 100.0, 0.0, 0.0, 2800.0])
            hi = np.array([1.0, 64.0, 100.0, 666.0, 128.0, 500.0, 100.0, 100.0, 6500.0])
        else:
            lo = np.array([p.min for p in param_specs.params])
            hi = np.array([p.max for p in param_specs.params])

        pop = lo + (hi - lo) * np.random.random((population_size, n_vars))
        scores = np.zeros(population_size)

        pop[0,:] = default

        for i in range(population_size):
            scores[i] = self.opt_f(pop[i])

            if verbose:
                print(f'Initializing population[{i}]: {scores[i]}')

        indices = np.arange(n_vars)

        # Here is the main evolutionary optimization loop
        for itr in range(n_iterations):
            worst = np.argmin(scores)
            best = np.argmax(scores)

            pop[worst,:] = pop[best,:]

            # Ensure at least one mutation
            n_mut = max(1, np.sum(np.random.random(n_vars) < mutation_rate))
            mut_indices = np.random.choice(indices, n_mut, replace=False)

            for i in mut_indices:
                pop[worst,i] = lo[i] + (hi[i] - lo[i]) * np.random.random()

            scores[worst] = self.opt_f(pop[worst])

            if verbose:
                print(f'Iteration {itr+1}: {scores[worst]}')

        best_i = np.argmax(scores)
        best = pop[best_i]

        if verbose:
            print(f'Optimized params with confidence score {scores[best_i]}:')
            print(best)

        msg = RealsenseSetParams([RealsenseParam(self.names[i], x) for i, x in enumerate(best)])
        self.param_pub.publish(msg)

        self.done = True

    def onStart(self):
        self.bridge = CvBridge()
        self.done = False
        self.thread = threading.Thread(target=self.run)
        self.thread.start()
        return True

    def execute(self):
        if not self.done:
            return self.step('Running...')

        self.thread.join()

        return self.success('Done')

class NVPExperiment(SkillDescription):
    def createDescription(self):
        self.addParam("Arm", Element("rparts:ArmDevice"), ParamTypes.Required)
        self.addParam("Reference", Element("skiros:TransformationPose"), ParamTypes.Required)
        self.addParam("Camera", Element("skiros:DepthCamera"), ParamTypes.Required)

        self.addParam("ViewFrame", Element("skiros:TransformationPose"), ParamTypes.Inferred)

        self.addPreCondition(self.getRelationCond("CHasV", "skiros:spatiallyRelated", "Camera", "ViewFrame", True))

class nvp_experiment(SkillBase):
    def createDescription(self):
        self.setDescription(NVPExperiment(), self.__class__.__name__)

    def expand(self, skill):
        skill.setProcessor(Loop())
        skill(
            self.skill("RandomLookAtMovement", "", specify={ "Distance": [0.5, 0.75], "Azimuth": [0.0, 45.0] }),

            #self.skill("Wait", "wait", specify={ "Duration": 5.0 }),

            self.skill("RealsenseAutoExposure", "", specify={ "Enabled": False }),
            self.skill("RealsenseAutoWhiteBalance", "", specify={ "Enabled": False }),
            self.skill("RealsenseResetParams", ""),
            self.skill("RealsenseDiscardFrames", "", specify={ "Number": 3 }),
            self.skill("RealsenseSaveRGB", "", specify={"OutputDir": '/home/simonklind/default/'}),
            self.skill("YOLOWriteToFile", "", specify={
                "Labels": ['apple', 'banana', 'scissors', 'fork', 'spoon', 'cup', 'mouse', 'book'],
                "File": "/home/simonklind/yolo_output_default.txt" }
            ),

            self.skill("RealsenseAutoExposure", "", specify={ "Enabled": True }),
            self.skill("RealsenseAutoWhiteBalance", "", specify={ "Enabled": True }),
            self.skill("Wait", "wait", specify={ "Duration": 1.0 }),
            self.skill("RealsenseSaveRGB", "", specify={"OutputDir": '/home/simonklind/auto/'}),
            self.skill("YOLOWriteToFile", "", specify={
                "Labels": ['apple', 'banana', 'scissors', 'fork', 'spoon', 'cup', 'mouse', 'book'],
                "File": "/home/simonklind/yolo_output_auto.txt" }
            ),

            self.skill("RealsenseAutoExposure", "", specify={ "Enabled": False }),
            self.skill("RealsenseAutoWhiteBalance", "", specify={ "Enabled": False }),
            self.skill("NVPOptimize", "", specify={
                "Population": 50,
                "MutationRate": 0.2,
                "NumIterations": 150,
                "UseGradImage": False,
                "Verbose": True,
            }),
            self.skill("RealsenseDiscardFrames", "", specify={ "Number": 3 }),
            self.skill("RealsenseSaveRGB", "", specify={"OutputDir": '/home/simonklind/opt/'}),
            self.skill("YOLOWriteToFile", "", specify={
                "Labels": ['apple', 'banana', 'scissors', 'fork', 'spoon', 'cup', 'mouse', 'book'],
                "File": "/home/simonklind/yolo_output_optimized.txt" }
            ),

            self.skill("RealsenseAutoExposure", "", specify={ "Enabled": False }),
            self.skill("RealsenseAutoWhiteBalance", "", specify={ "Enabled": False }),
            self.skill("NVPOptimize", "", specify={
                "Population": 50,
                "MutationRate": 0.2,
                "NumIterations": 150,
                "UseGradImage": True,
                "Verbose": True,
            }),
            self.skill("RealsenseDiscardFrames", "", specify={ "Number": 3 }),
            self.skill("RealsenseSaveRGB", "", specify={"OutputDir": '/home/simonklind/grad_opt/'}),
            self.skill("YOLOWriteToFile", "", specify={
                "Labels": ['apple', 'banana', 'scissors', 'fork', 'spoon', 'cup', 'mouse', 'book'],
                "File": "/home/simonklind/yolo_output_grad_optimized.txt" }
            ),
        )
