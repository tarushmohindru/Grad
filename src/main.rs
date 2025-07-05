use grad::tensor;
use grad::tensor::Tensor;

fn main() {
    let a = Tensor::new(vec![
        vec![vec![1., 2., 3.], vec![1., 2., 3.]],
        vec![vec![4., 5., 6.], vec![4., 5., 6.]],
    ]);

    let b = tensor![[[1., 2., 3.], [1., 2., 3.]], [[4., 5., 6.], [4., 5., 6.]]];
    let tensor = b + a;
    println!("{}", tensor);
}
