use burn::tensor::{
    Tensor,
    check_closeness,
};
use burn::backend::Cuda;

pub fn _closeness() {
    type Backend = Cuda;
    let device = Default::default();

    let tensor1 = Tensor::<Backend, 1>::from_floats(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.001, 7.002, 8.003, 9.004, 10.1],
        &device
    );

    let tensor2 = Tensor::<Backend, 1>::from_floats(
        [1.0, 2.0, 3.0, 4.000, 5.0, 6.0, 7.001, 8.002, 9.003, 10.004],
        &device
    );

    check_closeness(&tensor1, &tensor2);
}