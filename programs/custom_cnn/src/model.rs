use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{MaxPool2d, MaxPool2dConfig},
        PaddingConfig2d,
        Linear, LinearConfig,
        Relu,
    },
    prelude::*,
};


#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    conv4: Conv2d<B>,
    conv5: Conv2d<B>,
    conv6: Conv2d<B>,
    pool: MaxPool2d,
    lin1: Linear<B>,
    lin2: Linear<B>,
    activation: Relu,
}


#[derive(Config, Debug)]
pub struct ModelConfig {
    num_classes: usize,
    hidden_size: usize,
    #[config(default = "0.5")]
    dropout: f64,
}


impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            conv1: Conv2dConfig::new([1, 64], [3, 3]).with_padding(PaddingConfig2d::Same).init(device),
            conv2: Conv2dConfig::new([64, 64], [3, 3]).with_padding(PaddingConfig2d::Same).init(device),
            conv3: Conv2dConfig::new([64, 128], [3, 3]).with_padding(PaddingConfig2d::Same).init(device),
            conv4: Conv2dConfig::new([128, 128], [3, 3]).with_padding(PaddingConfig2d::Same).init(device),
            conv5: Conv2dConfig::new([128, 192], [3, 3]).with_padding(PaddingConfig2d::Same).init(device),
            conv6: Conv2dConfig::new([192, 192], [5, 5]).with_padding(PaddingConfig2d::Same).init(device),
            pool: MaxPool2dConfig::new([2, 2]).init(),
            lin1: LinearConfig::new(192 * 3 * 3, self.hidden_size).init(device),
            lin2: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
            activation: Relu::new(),
        }
    }
}


impl<B: Backend> Model<B> {
    /// # Shapes
    /// - Images [batch_size, height, width]
    /// - Output [batch_size, num_classes]
    pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = images.dims();

        // Create a channel at the second dimension.
        let x = images.reshape([batch_size, 1, height, width]);
        let x = self.activation.forward(self.conv1.forward(x));
        let x = self.activation.forward(self.conv2.forward(x));
        let x = self.activation.forward(self.conv3.forward(x));
        let x = self.pool.forward(x);

        let x = self.activation.forward(self.conv4.forward(x));
        let x = self.activation.forward(self.conv5.forward(x));
        let x = self.pool.forward(x);

        let x = self.activation.forward(self.conv6.forward(x));
        let x = self.pool.forward(x); // [batch_size, 16, 8, 8]
        
        let new_dim = x.dims(); // e.g. [64, 192, 3, 3]
        // dbg!(new_dim);
        let x = x.reshape([new_dim[0], new_dim[1] * new_dim[2] * new_dim[3]]);

        let x = self.activation.forward(self.lin1.forward(x));
        let x = self.lin2.forward(x);

        x
    }
}