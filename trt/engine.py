import tensorrt as trt
import numpy as np
from PIL import Image
import time
import pycuda.autoinit
import pycuda.driver as cuda


class TrtEngine(object):
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.context = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
        self.time_run = 0.0

    def test(self, nruns=1000):
        img = np.random.random((1, 3, 320, 320)).astype('float32')
        self.inputs[0]['host'] = np.ravel(img)
        # transfer data to the gpu
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        # run inference
        t_start = time.time()
        for i in range(nruns):
            self.context.execute_async_v2(
                bindings=self.bindings,
                stream_handle=self.stream.handle)
            # fetch outputs from gpu
            # for out in self.outputs:
            #     cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
            # synchronize stream
            # self.stream.synchronize()
        t = time.time() - t_start
        print("[test] {} runs, avg inference time: {:.8f} sec, fps: {:.2f}".format(nruns, t/nruns, 1/(t/nruns)))

    def infer(self, img):
        self.inputs[0]['host'] = np.ravel(img)
        # transfer data to the gpu
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        # run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        # fetch outputs from gpu
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        # synchronize stream
        self.stream.synchronize()
        data = [out['host'].copy() for out in self.outputs]
        return data

    def infer_all(self, imgs):
        """
        Inference images, 
        
        - Execution time calculation in this method include inference time and IO time, but
          speed test method 'self.test()' only consider the inference time.  
        """
        data = []
        t_start = time.time()
        for img in imgs:
            data.append(self.infer(img))
        t = time.time() - t_start
        print("[inference] {} samples, finished in {:4f} seconds, avg execution speed {:.4f}ms/img".format(
            len(imgs),
            t,
            t*1000/len(imgs)
        ))
        return data 


    