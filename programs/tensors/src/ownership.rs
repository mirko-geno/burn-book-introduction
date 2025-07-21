use burn::tensor::Tensor;
use burn::backend::Cuda;


fn _main() {
    type Backend = Cuda;
    let device = Default::default();
    /*
    Most tensor's operations take ownership of them. So 'clone' method is needed for reuse.
    There is no need to be worried about memory overhead because with cloning, the tensor's
    buﬀer isn't copied, and only a reference to it is increased.
    */
    let input = Tensor::<Backend, 1>::from_floats([1.0, 2.0, 3.0], &device);
    let min = input.clone().min();
    let max = input.clone().max();
    let input = (input.clone() - min.clone()).div(max - min);

    println!("{}", input.to_data()); // Success: [0.0, 0.33333334, 0.6666667, 1.0]
}