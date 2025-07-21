use burn::{
    backend::Cuda,
    data::dataset::Dataset,
};

fn main() {
    type Backend = Cuda<f32, i32>;

    let device = burn::backend::cuda::CudaDevice::default();
    let artifact_dir = "/tmp/guide";

    guide::inference::infer::<Backend>(
        artifact_dir,
        device,
        burn::data::dataset::vision::MnistDataset::test().get(42).unwrap()
    );
}
