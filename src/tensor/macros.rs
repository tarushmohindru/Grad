/// A macro for creating tensors with a convenient syntax
///
/// # Examples
///
/// ```rust
/// // 1D tensor (vector)
/// let t1 = tensor![1, 2, 3, 4];
///
/// // 2D tensor (matrix)
/// let t2 = tensor![
///     [1, 2, 3],
///     [4, 5, 6]
/// ];
///
/// // 3D tensor
/// let t3 = tensor![
///     [
///         [1, 2],
///         [3, 4]
///     ],
///     [
///         [5, 6],
///         [7, 8]
///     ]
/// ];
/// ```
#[macro_export]
macro_rules! tensor {
    // Entry point - convert the input and create tensor
    ($($tt:tt)*) => {
        $crate::tensor::Tensor::new($crate::tensor_vec![$($tt)*])
    };
}

/// Helper macro that converts array syntax to nested vectors
/// This is used internally by the tensor! macro
#[macro_export]
macro_rules! tensor_vec {
    // Handle multiple top-level elements separated by commas
    ($first:tt, $($rest:tt),* $(,)?) => {
        vec![$crate::tensor_vec![$first], $($crate::tensor_vec![$rest]),*]
    };

    // Handle array brackets - convert [a, b, c] to vec![...]
    ([$($inner:tt),* $(,)?]) => {
        vec![$($crate::tensor_vec![$inner]),*]
    };

    // Base case: literal values (numbers, etc.)
    ($item:expr) => {
        $item
    };
}

/// Re-export the tensor macro for easier access
pub use tensor;
pub use tensor_vec;
