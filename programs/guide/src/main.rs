mod model;
mod data;
mod training;
mod inference;

use crate::{
    model::ModelConfig,
    training::TrainingConfig
};
use burn::{
    backend::{Autodiff, Cuda}, data::dataset::Dataset, optim::AdamConfig
};

fn main() {
    type Backend = Cuda<f32, i32>;
    type AutodiffBackend = Autodiff<Backend>;

    let device = burn::backend::cuda::CudaDevice::default();
    let artifact_dir = "/tmp/guide";

    crate::training::train::<AutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device.clone(),
    );

    crate::inference::infer::<Backend>(
        artifact_dir,
        device,
        burn::data::dataset::vision::MnistDataset::test().get(42).unwrap()
    );
}
