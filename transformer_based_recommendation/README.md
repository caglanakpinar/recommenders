# Deep Conv. GAN - image generative AI

## How to install

we can run by using `poetry` environment. poetry will also install `recommenders` which whole model is ran by this ml ops framework.
```
poetry install .
```

## configurations
all configurable variables are available at `configs/params.yaml`.
all list of hyperparameters are available at `configs/params.yaml`.
there are some fields are mandatory and must be highligted here `configs/params.yaml`;
- `name`: this name will be used while naming captured images folder name. 
also, it will be in used while checkpoint saver folder as `training_checkpoint_<name>`
- `buffer_size`: number of images that will be used as input to GAN AI.
- `checkpoint_save_epoch`: example, 5, every five epochs, model will be saved at `training_checkpoint_<name>`. 
- `image_size`:
- `unit`: 
- `generator_layer_iterator`: this field will be used for how many `Conv2DTranspose` layer will be used in Generator model. 
This will be used in above code;
```
        models.py
        ....
        class Generator(BaseModel):
        ...
        self.iteration = params.get('generator_layer_iterator')
        self.divide = 2**params.get('generator_layer_iterator')
        ...
        for i, stride in zip(range(1, self.iteration+1), self.layer_strides):
            self.model.add(
                layers.Conv2DTranspose(
                    int(self.unit / (2**i)),
                    (self.kernel, self.kernel),
                    strides=(stride, stride),
                    padding='same',
                    use_bias=False
                )
            )
            self.batch_norm_and_l_relu()
        ...
```

- `discriminator_layer_iterator`: this field will be used for how many `Conv2D` layer will be used in Discriminator model. 
This will be used in above code;
```
        models.py
        ....
        class Discriminator(BaseModel):
        ...
        self.divide = 2**params.get('discriminator_layer_iterator')
        self.stride = params.get('discriminator_stride')
        ...
        def build(self):
            for i in range(self.iteration):
                self.model.add(
                    layers.Conv2D(
                        int(self.unit*(2**i)),
                        (self.kernel, self.kernel),
                        strides=(self.stride, self.stride),
                        padding='same',
                    )
                )
        ...
```
- `unit`: starter hidden unit for Generator model for the `Conv2DTranspose` and Discriminator model for `Conv2D`.
- `dropout_ratio`: this field will be used for Discriminator model at below lines;
```
    models.py
    ....
    class Discriminator(BaseModel):
    ...
    def l_relu_dropout(self):
        self.model.add(layers.LeakyReLU())
        self.model.add(layers.Dropout(self.dropout_ratio)) 
    ...
```

## How to Train

Training process will be executed by `dl_ml_ops` package. At `main.py`, `cli` is called from this package, 
and it is used below piece of code. You don't need to change anything on `main.py`
```
import ssl
from mlp.cli.cli import cli

ssl._create_default_https_context = ssl._create_unverified_context

if __name__ == "__main__":
    cli()
```

from `dl_ml_ops` there are some terminal argument to fill;
- `training_class`: `dcgan.DCGAN`. you can update this class if you needed. it is in `dcgan.py`
- `trainer_config_path`: `configs/params.yaml`. you can find it at `configs`.
- `continuous_training`: if True, model `dcgan.DCGAN` will be saved for every 5 epochs (this  5 epoch can be found at `configs/params.yaml` under `checkpoint_save_epoch`)


to train a model, use sample on terminal below;

```
poetry run python main.py \
model train \
--training_class dcgan.DCGAN \
--trainer_config_path configs/params.yaml \
--data_access_clas utils.CapturingImages \
--continuous_training True
```

While it is training after each epoch, Generator model will generate a image regarding the captured images. It will be stored under `training_checkpoints_<name>/ckpt/`.

## how to Tune Hyperparameters
To run hyperparameter tuning, below code will be executed from terminal
```
poetry run python main.py \
model train \
--tuning_class dcgan.HyperDCGAN \
--trainer_config_path configs/params.yaml \
--hyperparameter_config_path configs/hyper_params.yaml \
--data_access_clas utils.CapturingImages \
--continuous_training True
```
