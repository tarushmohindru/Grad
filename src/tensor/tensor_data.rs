use std::ops::{Add, Sub};

use crate::tensor::shape::{ElementOpp, Numeric};

#[derive(Debug, Clone)]
pub struct TensorData<T>(pub Vec<T>);

impl<T> Add for TensorData<T>
where
    T: ElementOpp<Output = T> + Clone,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.0.len(),
            rhs.0.len(),
            "Tensors must have same dimensions"
        );
        TensorData(
            self.0
                .into_iter()
                .zip(rhs.0.into_iter())
                .map(|(a, b)| a.element_add(b))
                .collect(),
        )
    }
}

impl<T> Sub for TensorData<T>
where
    T: ElementOpp<Output = T> + Clone,
{
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.0.len(),
            rhs.0.len(),
            "Tensors must have same dimensions"
        );
        TensorData(
            self.0
                .into_iter()
                .zip(rhs.0.into_iter())
                .map(|(a, b)| a.element_sub(b))
                .collect(),
        )
    }
}

impl<T> From<T> for TensorData<T>
where
    T: Numeric,
{
    fn from(scalar: T) -> Self {
        TensorData(vec![scalar])
    }
}
