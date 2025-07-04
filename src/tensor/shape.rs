/// Any element entering a ```Tensor``` must implement ```Shape```
pub trait Shape {
    fn shape(&self) -> Vec<usize>;
}

/// Trait for element-wise addition of tensor elements
pub trait ElementOpp<Rhs = Self> {
    type Output;
    fn element_add(self, rhs: Rhs) -> Self::Output;
    fn element_sub(self, rhs: Rhs) -> Self::Output;
}

impl<T> Shape for Vec<T>
where
    T: Shape,
{
    fn shape(&self) -> Vec<usize> {
        let mut shape = vec![self.len()];
        if let Some(first) = self.first() {
            shape.extend(first.shape());
        }
        shape
    }
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
}

use std::ops::{Add, Sub};

/// A helper trait for implementing ```Shape``` trait
pub trait Numeric {}

impl Numeric for u8 {}
impl Numeric for u16 {}
impl Numeric for u32 {}
impl Numeric for u64 {}
impl Numeric for u128 {}
impl Numeric for usize {}
impl Numeric for i8 {}
impl Numeric for i16 {}
impl Numeric for i32 {}
impl Numeric for i64 {}
impl Numeric for i128 {}
impl Numeric for isize {}
//impl Numeric for f16 {}
impl Numeric for f32 {}
impl Numeric for f64 {}
//impl Numeric for f128 {}

// Implement ElementAdd for all numeric types
impl<T> ElementOpp for T
where
    T: Numeric + Add<Output = T> + Sub<Output = T>,
{
    type Output = T;

    fn element_add(self, rhs: Self) -> Self::Output {
        self + rhs
    }

    fn element_sub(self, rhs: Self) -> Self::Output {
        self - rhs
    }
}

impl<T> Shape for T
where
    T: Numeric,
{
    fn shape(&self) -> Vec<usize> {
        vec![]
    }
}
