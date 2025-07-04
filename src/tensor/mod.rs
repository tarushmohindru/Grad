pub mod shape;
pub mod tensor_data;

use shape::{ElementAdd, Shape};
use std::{fmt::Debug, ops::Add};
use tensor_data::TensorData;

/// A type for holding matrix/vector data
#[derive(Debug)]
pub struct Tensor<T>
where
    T: Shape,
{
    pub dimension: Vec<usize>,
    pub data: TensorData<T>,
}

impl<T: Shape> Tensor<T> {
    pub fn new(data: Vec<T>) -> Self {
        let dim = data.shape();

        Self {
            dimension: dim,
            data: TensorData(data),
        }
    }
}

impl<T> Add for Tensor<T>
where
    T: Shape + ElementAdd<Output = T> + Clone,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.dimension, rhs.dimension,
            "Tensors must have same dimensions"
        );
        let data = self.data + rhs.data;

        Tensor::new(data.0)
    }
}

impl<T> core::fmt::Display for Tensor<T>
where
    T: Shape + Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.data.0)
    }
}
