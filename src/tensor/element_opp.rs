use std::ops::{Add, Mul, Sub};

use crate::tensor::shape::Numeric;

/// Trait for element-wise operations of tensor elements
pub trait ElementOpp<Rhs = Self> {
    type Output;
    fn element_add(self, rhs: Rhs) -> Self::Output;
    fn element_sub(self, rhs: Rhs) -> Self::Output;
    fn element_mul(self, rhs: Rhs) -> Self::Output;
}

// Implementation of ElementAdd for Vec<T>
impl<T> ElementOpp for Vec<T>
where
    T: ElementOpp<Output = T> + Clone,
{
    type Output = Vec<T>;

    fn element_add(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.len(),
            rhs.len(),
            "Vectors must have same length for addition"
        );

        self.into_iter()
            .zip(rhs.into_iter())
            .map(|(a, b)| a.element_add(b))
            .collect()
    }

    fn element_sub(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.len(),
            rhs.len(),
            "Vectors must have same length for addition"
        );

        self.into_iter()
            .zip(rhs.into_iter())
            .map(|(a, b)| a.element_sub(b))
            .collect()
    }

    fn element_mul(self, rhs: Self) -> Self::Output {
        self.into_iter()
            .zip(rhs.into_iter())
            .map(|(a, b)| a.element_mul(b))
            .collect()
    }
}

impl<T> ElementOpp for T
where
    T: Numeric + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    type Output = T;

    fn element_add(self, rhs: Self) -> Self::Output {
        self + rhs
    }

    fn element_sub(self, rhs: Self) -> Self::Output {
        self - rhs
    }

    fn element_mul(self, rhs: Self) -> Self::Output {
        self * rhs
    }
}
