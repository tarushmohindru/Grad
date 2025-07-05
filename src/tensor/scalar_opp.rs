use std::ops::{Div, Mul};

use crate::tensor::shape::Numeric;

/// Trait for supporting ```scalar``` * ```Tensor``` multiplication
pub trait ScalarOpp<S> {
    type Output;
    fn scalar_mul(self, scalar: S) -> Self::Output;
    fn scalar_div(self, scalar: S) -> Self::Output;
}

impl<S, T> ScalarOpp<S> for T
where
    T: Numeric + Mul<S, Output = T> + Div<S, Output = T>,
    S: Numeric + Copy,
{
    type Output = Self;

    fn scalar_mul(self, scalar: S) -> Self::Output {
        self * scalar
    }

    fn scalar_div(self, scalar: S) -> Self::Output {
        self / scalar
    }
}

impl<S, T> ScalarOpp<S> for Vec<T>
where
    T: ScalarOpp<S, Output = T> + Clone,
    S: Numeric + Copy,
{
    type Output = Self;
    fn scalar_mul(self, scalar: S) -> Self::Output {
        self.into_iter().map(|a| a.scalar_mul(scalar)).collect()
    }

    fn scalar_div(self, scalar: S) -> Self::Output {
        self.into_iter().map(|a| a.scalar_div(scalar)).collect()
    }
}
