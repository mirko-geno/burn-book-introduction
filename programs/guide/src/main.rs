mod model;
mod data;

use crate::model::ModelConfig;
use burn::backend::Cuda;

fn main() {
    type Backend = Cuda<f32, i32>;

    let device = Default::default();
    let model = ModelConfig::new(10, 512).init::<Backend>(&device);

    println!("{model}");


}
