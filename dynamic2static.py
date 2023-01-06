import paddle
import argparse
from paddle.static import InputSpec
 
 
def main(args):
    if args.model_tag == 'bsacloudnet-lite':
        from models.bsacloudnet_lite import BSACloudNet
        model = BSACloudNet(mode='infer')
    elif args.model_tag == 'bsacloudnet':
        from models.bsacloudnet import BSACloudNet
        model = BSACloudNet(mode='infer')
    elif args.model_tag == 'bsacloudnet-large':
        from models.bsacloudnet_large import BSACloudNet
        model = BSACloudNet(mode='infer')
    else:
        raise Exception("Model not found")

    model.set_state_dict(paddle.load(args.ckpt_path))
    model.eval()
	
    x_spec = InputSpec(shape=[1, 3, 320, 320], dtype='float32', name='x')
    model = paddle.jit.save(model, path=args.save_dir+"/"+args.model_tag, input_spec=[x_spec])
 
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_tag', type=str, default='bsacloudnet-lite', 
        help="the tag of model (default: bsacloudnet-lite)")
    parser.add_argument(
        '--ckpt_path', type=str, default="./ckpts/bsacloudnet-lite_epochs_100.pdparam", 
        help="path of checkpoint (default: ./ckpts/bsacloudnet-lite_epochs_100.pdparam)")
    parser.add_argument(
        '--save_dir', type=str, default="./ckpts/static", 
        help="dir of saved static checkpoints (default: ./ckpts/static)")


    args = parser.parse_args()
    main(args)


