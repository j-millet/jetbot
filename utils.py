import os
import pandas as pd
import torch
from torchvision.transforms import v2


def get_data(path: str = 'dataset') -> pd.DataFrame:
    """
    Reads the dataset from the specified path and returns a DataFrame.
    """
    filepaths = os.listdir(path)
    csv_filepaths = [path for path in filepaths if path.endswith('.csv')]

    dataframes = []
    for filepath in csv_filepaths:
        df = pd.read_csv(
            os.path.join(path, filepath),
            names = ['image_number', 'forward_signal', 'left_signal']
        )
        df.insert(0, 'image_path', filepath[:-4])
        df['image_path'] = df.apply(
            lambda x: x['image_path'] + '/' + ('000' + str(x['image_number']))[-4:] + '.jpg',
            axis=1,
        )
        dataframes.append(df)

    return pd.concat(dataframes, axis=0)


def get_transorms() -> tuple:
    """
    Returns the training and validation transformations for the dataset.
    """
    add_noise = v2.Lambda(lambda img: img + torch.randn_like(img) * 0.02)

    train_transforms = v2.Compose([
        v2.RandomResizedCrop(
            size=(224, 224),
            scale=(0.9, 1.0),
            ratio=(3/4, 4/3)
        ),
        v2.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.01
        ),
        v2.RandomPerspective(
            distortion_scale=0.1,
            p=0.5
        ),
        v2.RandomApply(
            transforms=[v2.GaussianBlur(kernel_size=(3,3), sigma=(0.1, 2.0))],
            p=0.3
        ),
        v2.ToTensor(),
        v2.RandomErasing(
            p=0.5,
            scale=(0.02, 0.1),
            ratio=(0.3, 3.3),
        ),
        v2.RandomApply([add_noise], p=0.5),
    ])
    val_transforms = v2.Compose([
        v2.ToTensor(),
    ])
    return (train_transforms, val_transforms)


def save_onnx_model(model: torch.nn.Module, name: str) -> None:
    """
    Placeholder function to save the ONNX model.
    """
    example_inputs = (torch.randn(1, 3, 224, 224),)
    onnx_program = torch.onnx.export(model, example_inputs, dynamo=True)
    onnx_program.optimize()
    onnx_program.save(f"/saved_models/{name}.onnx")
    print(f"Model saved as {name}.onnx")