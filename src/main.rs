use grad::tensor::Tensor;

fn main() {
    let a = vec![
        vec![vec![1, 2, 3], vec![1, 2, 3]],
        vec![vec![5, 6, 7], vec![5, 6, 7]],
    ];
    let tensor = Tensor::new(a.clone()) * 2;
    println!("{}", tensor);
}
