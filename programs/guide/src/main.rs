mod model;
mod data;
mod training;

use crate::{
    model::ModelConfig,
    training::{TrainingConfig, train}
};
use burn::{
    backend::{Autodiff, Cuda},
    optim::AdamConfig,
};

fn main() {
    type Backend = Cuda<f32, i32>;
    type AutodiffBackend = Autodiff<Backend>;

    let device = burn::backend::cuda::CudaDevice::default();
    let artifact_dir = "/tmp/guide";

    train::<AutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device.clone(),
    );
}
