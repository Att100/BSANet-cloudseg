import paddle
import argparse
from paddle.static import InputSpec
 
 
def main(args):
    if args.model_tag == 'BSANet-lite':
        from models.bsanet_lite import BSANet
        model = BSANet(mode='infer')
    elif args.model_tag == 'BSANet':
        from models.bsanet import BSANet
        model = BSANet(mode='infer')
    elif args.model_tag == 'BSANet-large':
        from models.bsanet_large import BSANet
        model = BSANet(mode='infer')
    else:
        raise Exception("Model not found")

    model.set_state_dict(paddle.load(args.ckpt_path))
    model.eval()
	
    x_spec = InputSpec(shape=[1, 3, 320, 320], dtype='float32', name='x')
    model = paddle.jit.save(model, path=args.save_dir+"/"+args.model_tag, input_spec=[x_spec])
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_tag', type=str, default='BSANet-lite', 
        help="the tag of model (default: BSANet-lite)")
    parser.add_argument(
        '--ckpt_path', type=str, default="./ckpts/BSANet-lite_epochs_100.pdparam", 
        help="path of checkpoint (default: ./ckpts/BSANet-lite_epochs_100.pdparam)")
    parser.add_argument(
        '--save_dir', type=str, default="./ckpts/static", 
        help="dir of saved static checkpoints (default: ./ckpts/static)")


    args = parser.parse_args()
    main(args)


