use burn::tensor::{
    Tensor,
    set_print_options,
    PrintOptions,
};
use burn::backend::{
    Cuda,
};

fn _main() {
    type Backend = Cuda;
    let device = Default::default();

    let tensor = Tensor::<Backend, 2>::full([2, 3], 0.123456789, &device);
    println!("{}", tensor);

    // To limit number of decimals:
    println!("{:.2}", tensor);

    let print_options = PrintOptions {
        precision: Some(2),
        threshold: Default::default(),
        edge_items: Default::default(),
    };

    set_print_options(print_options);
}