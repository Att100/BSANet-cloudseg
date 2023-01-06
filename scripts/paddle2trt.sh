python dynamic2static.py
paddle2onnx --model_dir ./ckpts/static --model_filename bsamnet_lite.pdmodel --params_filename bsamnet_lite.pdiparams --save_file ./ckpts/onnx/bsamnet_lite.onnx
python onnx2trt.py