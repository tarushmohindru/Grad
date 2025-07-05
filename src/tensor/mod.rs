pub mod element_opp;
pub mod scalar_opp;
pub mod shape;
pub mod tensor_data;

use element_opp::ElementOpp;
use shape::Shape;
use std::{
    fmt::Debug,
    ops::{Add, Div, Mul, Sub},
};
use tensor_data::TensorData;

use crate::tensor::{scalar_opp::ScalarOpp, shape::Numeric};

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
    T: Shape + ElementOpp<Output = T> + Clone,
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

impl<T> Sub for Tensor<T>
where
    T: Shape + ElementOpp<Output = T> + Clone,
{
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.dimension, rhs.dimension,
            "Tensors must have same dimensions"
        );
        let data = self.data - rhs.data;
        Tensor::new(data.0)
    }
}

impl<T> Mul for Tensor<T>
where
    T: Shape + ElementOpp<Output = T> + Clone,
{
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.dimension, rhs.dimension,
            "Tensers must have same dimension"
        );
        let data = self.data * rhs.data;
        Tensor::new(data.0)
    }
}

impl<T> Div for Tensor<T>
where
    T: Shape + ElementOpp<Output = T> + Clone,
{
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.dimension, rhs.dimension,
            "Tensers must have same dimension"
        );
        let data = self.data / rhs.data;
        Tensor::new(data.0)
    }
}

impl<T, S> Mul<S> for Tensor<T>
where
    T: Shape + Clone + ScalarOpp<S, Output = T>,
    S: Numeric + Copy,
{
    type Output = Self;
    fn mul(self, scalar: S) -> Self::Output {
        let new_data = self
            .data
            .0
            .into_iter()
            .map(|x| x.scalar_mul(scalar))
            .collect();
        Tensor::new(new_data)
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
