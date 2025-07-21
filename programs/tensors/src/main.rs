mod initialization;
mod ownership;
mod display;
mod closeness;

use burn::tensor::Tensor;
use burn::backend::Cuda;

// Type alias for the backend to use.
type Backend = Cuda;

fn main() {
    let device = Default::default();

    /*
    Burn Tensors are defined by the number of dimensions D in its declaration as opposed to its shape.
    The actual shape of the tensor is inferred from its initialization.
    */
    let floats = [1., 2., 3., 4., 5.];
    let _tensor_0 = Tensor::<Backend, 1>::from_floats(floats, &device);

    // Creation of two tensors, the first with explicit values and the second one with ones, with the same shape as the first
    let tensor_1 = Tensor::<Backend, 2>::from_data([[2., 3.], [4., 5.]], &device);
    let tensor_2 = Tensor::<Backend, 2>::ones_like(&tensor_1);

    // Print the element-wise addition (done with the CUDA backend) of the two tensors.
    println!("{}", tensor_1 + tensor_2);
}
