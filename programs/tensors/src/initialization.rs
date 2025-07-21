// This code shows multiple ways to initialize tensors
use burn::tensor::{Tensor, TensorData, Int};
use burn::backend::{
    Cuda,
    cuda::CudaDevice,
};

struct _BodyMetrics {
    age: i8,
    height: i16,
    weight: f32,
}

fn _main() {
    type Backend = Cuda;
    let device = CudaDevice::default();

    let _tensor_0 = Tensor::<Backend, 1>::from_data(TensorData::from([1.0, 2.0, 3.0]), &device);
    
    // Using TensorData is not necessary since it is automatically converted:
    let _tensor_1 = Tensor::<Backend, 1>::from_data([1.0, 2.0, 3.0], &device);
    
    // Initialization using from_floats (Recommended for f32 ElementType)
    let _tensor_3 = Tensor::<Backend, 1>::from_floats([1.0, 2.0, 3.0], &device);

    // Initialization of Int Tensor from array slices
    let arr: [i32; 6] = [1, 2, 3, 4, 5, 6];
    let _tensor_4 = Tensor::<Backend, 1, Int>::from_data(TensorData::from(&arr[0..3]), &device);

    // Init from custom type
    let bmi = _BodyMetrics {
        age: 25,
        height: 180,
        weight: 80.0,
    };

    let data = TensorData::from([
        bmi.age as f32,
        bmi.height as f32,
        bmi.weight
    ]);

    let _tensor_5 = Tensor::<Backend, 1>::from_data(data, &device);
}