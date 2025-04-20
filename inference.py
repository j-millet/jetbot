import argparse
import torch

import onnx
import onnxruntime

from utils import get_transforms, to_numpy

def main(args):
    _, val_t = get_transforms()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = onnx.load(args.model_file)
    onnx.checker.check_model(model)
    model.to(device)
    model.eval()
    
    ort_session = onnxruntime.InferenceSession(args.model_file, providers=["CPUExecutionProvider"])
    
    datastream = None

    while datastream:
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(datastream)}
        ort_outs = ort_session.run(None, ort_inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a JetBot steering model."
    )
    parser.add_argument(
        "model_file",
        help="Path to your saved ONNX model."
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="batch size for training/validation"
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="number of epochs to train"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="learning rate"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="validation split fraction"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="random seed"
    )
    args = parser.parse_args()
    main(args)