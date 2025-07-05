/// Any element entering a ```Tensor``` must implement ```Shape```
pub trait Shape {
    fn shape(&self) -> Vec<usize>;
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

impl<T> Shape for T
where
    T: Numeric,
{
    fn shape(&self) -> Vec<usize> {
        vec![]
    }
}
