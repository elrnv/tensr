#![macro_use]

use flatk::*;
use std::borrow::Borrow;
use std::iter;

// The goal is to achieve this type of notation (from taco):
//  let A: Tensor<f64> = /* Constructor */;
//  IndexVar i, j, k;
//  A(i,j) = B(i,j,k) * c(k);

// Construct a tensor from a given size
trait FromSize: Sized {
    /// Construct the Tensor from a collection of sizes.
    #[inline]
    fn from_size<I: IntoIterator>(size: I) -> Self
    where
        I::Item: Borrow<usize>,
    {
        Self::from_size_iter(size.into_iter().map(|i| *i.borrow()))
    }

    /// Construct the Tensor from a size iterator.
    fn from_size_iter<I: Iterator<Item = usize>>(size: I) -> Self;
}

impl<S: Set + FromSize> FromSize for Sparse<S> {
    fn from_size_iter<I: Iterator<Item = usize>>(mut size: I) -> Self {
        Self::from_dim(
            vec![],
            size.next().expect("Not enough sizes specified"),
            S::from_size_iter(iter::once(0).chain(size)),
        )
    }
}
impl<S: Set + FromSize> FromSize for Chunked<S> {
    fn from_size_iter<I: Iterator<Item = usize>>(mut size: I) -> Self {
        let n = size.next().expect("Not enough sizes specified");
        Self::from_offsets(vec![0; n + 1], S::from_size_iter(size))
    }
}
impl<S: Set + FromSize> FromSize for ChunkedN<S> {
    fn from_size_iter<I: Iterator<Item = usize>>(mut size: I) -> Self {
        let rows = size.next().expect("Not enough sizes specified");
        let cols = size.next().expect("Not enough sizes specified");
        Self::from_flat_with_stride(cols, S::from_size_iter(iter::once(rows * cols).chain(size)))
    }
}

impl<S: Set + FromSize, N: Unsigned + Default> FromSize for UniChunked<S, U<N>> {
    fn from_size_iter<I: Iterator<Item = usize>>(mut size: I) -> Self {
        let rows = size.next().expect("Not enough sizes specified");
        let cols = size.next().expect("Not enough sizes specified");
        assert_eq!(
            cols,
            N::to_usize(),
            "Static size doesn't correspond to the one given"
        );
        Self::from_flat(S::from_size_iter(iter::once(rows * cols).chain(size)))
    }
}

impl<T: Default + Clone> FromSize for Vec<T> {
    fn from_size_iter<I: Iterator<Item = usize>>(mut size: I) -> Self {
        vec![T::default(); size.next().expect("Not enough sizes specified")]
    }
}

macro_rules! impl_from_size_for_array {
    () => { };
    ($n:literal $(,$ns:literal)* $(,)*) => {
        impl<T: Default + Copy> FromSize for [T; $n] {
            fn from_size_iter<I: Iterator<Item = usize>>(mut size: I) -> Self {
                let n = size.next().expect("Not enough sizes specified");
                assert_eq!( n, $n, "Static size doesn't correspond to the one given");
                [T::default(); $n]
            }
        }
        impl_from_size_for_array!($($ns,)*);
    }
}

impl_from_size_for_array!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

#[macro_export]
macro_rules! Tensor {
    // Base cases
    ($real:ident $(; @)?) => {
        Vec<$real>
    };
    // Starters
    // Outermost collections don't need to be chunked.
    ($(($l:expr))? $real:ident; S $($layout:tt)*) => {
        Sparse<Tensor![$real; @ $($layout)*]>
    };
    ($(($l:expr))? $real:ident; D $($layout:tt)*) => {
        Tensor![$real; @ $($layout)*]
    };
    ($(($l:expr))? $real:ident; @ S $($layout:tt)*) => {
        Chunked<Sparse<Tensor![$real; @ $($layout)*]>>
    };
    ($(($l:expr))? $real:ident; @ D $($layout:tt)*) => {
        ChunkedN<Tensor![$real; @ $($layout)*]>
    };
    (($l:expr) $real:ident; @ $n:expr) => {
        [$real; $l * $n]
    };
    // TODO: Turn these into a macro
    // See: https://github.com/rust-lang/rust/issues/35853 for reference
    ($(($l:expr))? $real:ident; @ 1 $($layout:tt)*) => {
        UniChunked<Tensor![$(($l * 1))? $real; @ $($layout)*], U1>
    };
    ($(($l:expr))? $real:ident; @ 2 $($layout:tt)*) => {
        UniChunked<Tensor![$(($l * 2))? $real; @ $($layout)*], U2>
    };
    ($(($l:expr))? $real:ident; @ 3 $($layout:tt)*) => {
        UniChunked<Tensor![$(($l * 3))? $real; @ $($layout)*], U3>
    };
    ($(($l:expr))? $real:ident; @ 4 $($layout:tt)*) => {
        UniChunked<Tensor![$(($l * 4))? $real; @ $($layout)*], U4>
    };
    ($(($l:expr))? $real:ident; @ 5 $($layout:tt)*) => {
        UniChunked<Tensor![$(($l * 5))? $real; @ $($layout)*], U5>
    };
    ($(($l:expr))? $real:ident; @ 6 $($layout:tt)*) => {
        UniChunked<Tensor![$(($l * 6))? $real; @ $($layout)*], U6>
    };
    ($(($l:expr))? $real:ident; @ 7 $($layout:tt)*) => {
        UniChunked<Tensor![$(($l * 7))? $real; @ $($layout)*], U7>
    };
    ($(($l:expr))? $real:ident; @ 8 $($layout:tt)*) => {
        UniChunked<Tensor![$(($l * 8))? $real; @ $($layout)*], U8>
    };
    ($(($l:expr))? $real:ident; @ 9 $($layout:tt)*) => {
        UniChunked<Tensor![$(($l * 9))? $real; @ $($layout)*], U9>
    };
    ($(($l:expr))? $real:ident; @ 10 $($layout:tt)*) => {
        UniChunked<Tensor![$(($l * 10))? $real; @ $($layout)*], U10>
    };
    ($(($l:expr))? $real:ident; @ 11 $($layout:tt)*) => {
        UniChunked<Tensor![$(($l * 11))? $real; @ $($layout)*], U11>
    };
    ($(($l:expr))? $real:ident; @ 12 $($layout:tt)*) => {
        UniChunked<Tensor![$(($l * 12))? $real; @ $($layout)*], U12>
    };
    ($(($l:expr))? $real:ident; @ 13 $($layout:tt)*) => {
        UniChunked<Tensor![$(($l * 13))? $real; @ $($layout)*], U13>
    };
    ($(($l:expr))? $real:ident; @ 14 $($layout:tt)*) => {
        UniChunked<Tensor![$(($l * 14))? $real; @ $($layout)*], U14>
    };
    ($(($l:expr))? $real:ident; @ 15 $($layout:tt)*) => {
        UniChunked<Tensor![$(($l * 15))? $real; @ $($layout)*], U15>
    };
    ($(($l:expr))? $real:ident; @ 16 $($layout:tt)*) => {
        UniChunked<Tensor![$(($l * 16))? $real; @ $($layout)*], U16>
    };
    ($(($l:expr))? $real:ident; $($layout:tt)*) => {
        Tensor![(1) $real; @ $($layout)*]
    };
}

macro_rules! expr {
    ($) => {};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create() {
        // Sparse vector
        let s = <Tensor![f64; S]>::from_size(&[512]);
        assert_eq!(Sparse::from_dim(vec![], 512, vec![]), s);

        // Sparse row dense column matrix
        let sd = <Tensor![f64; S D]>::from_size(&[64, 32]);
        assert_eq!(
            &Sparse::from_dim(vec![], 64, ChunkedN::from_flat_with_stride(32, vec![])),
            &sd
        );

        // Sparse row dense column matrix of 3D vectors
        let sd3 = <Tensor![f64; S D 3]>::from_size(&[512, 64, 3]);
        assert_eq!(
            Sparse::from_dim(
                vec![],
                512,
                ChunkedN::from_flat_with_stride(64, Chunked3::from_flat(vec![]))
            ),
            sd3
        );

        // Standard Dense matrix
        let dd = <Tensor![f64; D D]>::from_size(vec![128, 64]);
        assert_eq!(ChunkedN::from_flat_with_stride(64, vec![0.0; 128 * 64]), dd);

        // Dense row sparse column (CSR) matrix
        let ds = <Tensor![f64; D S]>::from_size(&[32, 12]);
        assert_eq!(
            Chunked::from_offsets(vec![0; 32 + 1], Sparse::from_dim(vec![], 12, vec![])),
            ds
        );

        // Dense blocks of sparse row dense col matrices
        let dsd = <Tensor![f64; D S D]>::from_size(&[16, 64, 32]);
        assert_eq!(Chunked::from_offsets(vec![0; 16 + 1], sd.clone()), dsd);

        // Dense row sparse column matrix of DS blocks
        let dsds = <Tensor![f64; D S D S]>::from_size(&[16, 64, 32, 12]);
        assert_eq!(
            Chunked::from_offsets(
                vec![0; 16 + 1],
                Sparse::from_dim(
                    vec![],
                    64,
                    ChunkedN::from_flat_with_stride(
                        32,
                        Chunked::from_offsets(vec![0], Sparse::from_dim(vec![], 12, vec![]))
                    )
                )
            ),
            dsds
        );

        // Sparse row dense column matrix of SD blocks
        let sdsd = <Tensor![f64; S D S D]>::from_size(&[64, 32, 12, 16]);
        assert_eq!(
            Sparse::from_dim(
                vec![],
                64,
                ChunkedN::from_flat_with_stride(
                    32,
                    Chunked::from_offsets(
                        vec![0],
                        Sparse::from_dim(vec![], 12, ChunkedN::from_flat_with_stride(16, vec![]))
                    )
                )
            ),
            sdsd
        );

        // Sparse row sparse column matrix of SS blocks
        let ssss = <Tensor![f64; S S S S]>::from_size(&[64, 32, 12, 16]);
        assert_eq!(
            Sparse::from_dim(
                vec![],
                64,
                Chunked::from_offsets(
                    vec![0],
                    Sparse::from_dim(
                        vec![],
                        32,
                        Chunked::from_offsets(
                            vec![0],
                            Sparse::from_dim(
                                vec![],
                                12,
                                Chunked::from_offsets(
                                    vec![0],
                                    Sparse::from_dim(vec![], 16, vec![])
                                )
                            )
                        )
                    )
                )
            ),
            ssss
        );

        // Sparse row sparse column matrix
        let ss = <Tensor![f64; S S]>::from_size(&[128, 64]);
        assert_eq!(
            Sparse::from_dim(
                vec![],
                128,
                Chunked::from_offsets(vec![0], Sparse::from_dim(vec![], 64, vec![]))
            ),
            ss
        );

        // Sparse row with 3 vector columns
        let s3 = <Tensor![f64; S 3]>::from_size(&[128, 3]);
        assert_eq!(
            Sparse::from_dim(vec![], 128, Chunked3::from_flat(vec![])),
            s3
        );

        // Dense row with 3 vector columns
        let d3 = <Tensor![f64; D 3]>::from_size(&[128, 3]);
        assert_eq!(Chunked3::from_flat(vec![0.0; 128 * 3]), d3);

        // A 3D vector
        let v3 = <Tensor![f64; 3]>::from_size(&[3]);
        assert_eq!([0.0; 3], v3);

        // A 3x3 matrix
        let m33 = <Tensor![f64; 3 3]>::from_size(&[3, 3]);
        assert_eq!(Chunked3::from_flat([0.0; 9]), m33);
    }

    #[test]
    fn multiply() {
        struct I;
        struct J;
        struct K;
        struct L;
        let c = <Tensor![f64; S]>::from_size(&[512]);
        let b = <Tensor![f64; D S S 3]>::from_size(&[64, 42, 512, 3]);
        let a: Tensor![f64; D S 3] = (b.expr::<I, J, K, L>() * c.expr::<K>()).eval::<I, J, L>();
    }
}
