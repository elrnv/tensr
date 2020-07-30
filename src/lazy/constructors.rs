#![macro_use]

use std::borrow::Borrow;
use std::iter;

use flatk::*;

// The goal is to achieve this type of notation (from taco):
//  let A: Tensor<f64> = /* Constructor */;
//  IndexVar i, j, k;
//  A(i,j) = B(i,j,k) * c(k);

// Construct a tensor from a given size
pub trait FromSize: Sized {
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
    ($real:ident; & $($l:lifetime)? $(@)?) => {
        & $($l)? [$real]
    };
    ($real:ident; & $($l:lifetime)? mut $(@)?) => {
        & $($l)? mut [$real]
    };

    // Outermost collections don't need to be chunked.
    // Sparse starter
    ($(($n:expr))? $real:ident; S $($layout:tt)*) => {
        Sparse<Tensor![$real; @ $($layout)*]>
    };
    ($(($n:expr))? $real:ident; & $($l:lifetime)? S $($layout:tt)*) => {
        Sparse<Tensor![$real; & $($l)? @ $($layout)*], ::std::ops::RangeTo<usize>, & $($l)? [usize]>
    };
    ($(($n:expr))? $real:ident; & $($l:lifetime)? mut S $($layout:tt)*) => {
        Sparse<Tensor![$real; & $($l)? mut @ $($layout)*], ::std::ops::RangeTo<usize>, & $($l)? mut [usize]>
    };

    // Dense starter
    ($(($n:expr))? $real:ident; $(& $($l:lifetime)? $($mut:ident)?)? D $($layout:tt)*) => {
        Tensor![$real; $(& $($l)? $($mut)?)? @ $($layout)*]
    };

    // Sparse recursive
    ($(($n:expr))? $real:ident; @ S $($layout:tt)*) => {
        Chunked<Sparse<Tensor![$real; @ $($layout)*]>>
    };
    ($(($n:expr))? $real:ident; & $($l:lifetime)? @ S $($layout:tt)*) => {
        Chunked<Sparse<Tensor![$real; & $($l)? @ $($layout)*], ::std::ops::RangeTo<usize>, & $($l)? [usize]>, Offsets<&$($l)? [usize]>>
    };
    ($(($n:expr))? $real:ident; & $($l:lifetime)? mut @ S $($layout:tt)*) => {
        Chunked<Sparse<Tensor![$real; & $($l)? mut @ $($layout)*], ::std::ops::RangeTo<usize>, & $($l)? mut [usize]>, Offsets<& $($l)? mut [usize]>>
    };

    // Dense recursive
    ($(($n:expr))? $real:ident; $(& $($l:lifetime)? $($mut:ident)?)? @ D $($layout:tt)*) => {
        ChunkedN<Tensor![$real; $(& $($l)? $($mut)?)? @ $($layout)*]>
    };

    // Array base case
    (($n:expr) $real:ident; $(& $($l:lifetime)? $($mut:ident)?)? @ $m:expr) => {
        $(& $($l)? $($mut)?)? [$real; $n * $m]
    };

    // Uniform dense recursive case
    // TODO: Turn these into a macro
    // See: https://github.com/rust-lang/rust/issues/35853 for reference
    ($(($n:expr))? $real:ident; $(& $($l:lifetime)? $($mut:ident)?)? @ 1 $($layout:tt)*) => {
        UniChunked<Tensor![$(($n * 1))? $real; $(& $($l)? $($mut)?)? @ $($layout)*], U1>
    };
    ($(($n:expr))? $real:ident; $(& $($l:lifetime)? $($mut:ident)?)? @ 2 $($layout:tt)*) => {
        UniChunked<Tensor![$(($n * 2))? $real; $(& $($l)? $($mut)?)? @ $($layout)*], U2>
    };
    ($(($n:expr))? $real:ident; $(& $($l:lifetime)? $($mut:ident)?)? @ 3 $($layout:tt)*) => {
        UniChunked<Tensor![$(($n * 3))? $real; $(& $($l)? $($mut)?)? @ $($layout)*], U3>
    };
    ($(($n:expr))? $real:ident; $(& $($l:lifetime)? $($mut:ident)?)? @ 4 $($layout:tt)*) => {
        UniChunked<Tensor![$(($n * 4))? $real; $(& $($l)? $($mut)?)? @ $($layout)*], U4>
    };
    ($(($n:expr))? $real:ident; $(& $($l:lifetime)? $($mut:ident)?)? @ 5 $($layout:tt)*) => {
        UniChunked<Tensor![$(($n * 5))? $real; $(& $($l)? $($mut)?)? @ $($layout)*], U5>
    };
    ($(($n:expr))? $real:ident; $(& $($l:lifetime)? $($mut:ident)?)? @ 6 $($layout:tt)*) => {
        UniChunked<Tensor![$(($n * 6))? $real; $(& $($l)? $($mut)?)? @ $($layout)*], U6>
    };
    ($(($n:expr))? $real:ident; $(& $($l:lifetime)? $($mut:ident)?)? @ 7 $($layout:tt)*) => {
        UniChunked<Tensor![$(($n * 7))? $real; $(& $($l)? $($mut)?)? @ $($layout)*], U7>
    };
    ($(($n:expr))? $real:ident; $(& $($l:lifetime)? $($mut:ident)?)? @ 8 $($layout:tt)*) => {
        UniChunked<Tensor![$(($n * 8))? $real; $(& $($l)? $($mut)?)? @ $($layout)*], U8>
    };
    ($(($n:expr))? $real:ident; $(& $($l:lifetime)? $($mut:ident)?)? @ 9 $($layout:tt)*) => {
        UniChunked<Tensor![$(($n * 9))? $real; $(& $($l)? $($mut)?)? @ $($layout)*], U9>
    };
    ($(($n:expr))? $real:ident; $(& $($l:lifetime)? $($mut:ident)?)? @ 10 $($layout:tt)*) => {
        UniChunked<Tensor![$(($n * 10))? $real; $(& $($l)? $($mut)?)? @ $($layout)*], U10>
    };
    ($(($n:expr))? $real:ident; $(& $($l:lifetime)? $($mut:ident)?)? @ 11 $($layout:tt)*) => {
        UniChunked<Tensor![$(($n * 11))? $real; $(& $($l)? $($mut)?)? @ $($layout)*], U11>
    };
    ($(($n:expr))? $real:ident; $(& $($l:lifetime)? $($mut:ident)?)? @ 12 $($layout:tt)*) => {
        UniChunked<Tensor![$(($n * 12))? $real; $(& $($l)? $($mut)?)? @ $($layout)*], U12>
    };
    ($(($n:expr))? $real:ident; $(& $($l:lifetime)? $($mut:ident)?)? @ 13 $($layout:tt)*) => {
        UniChunked<Tensor![$(($n * 13))? $real; $(& $($l)? $($mut)?)? @ $($layout)*], U13>
    };
    ($(($n:expr))? $real:ident; $(& $($l:lifetime)? $($mut:ident)?)? @ 14 $($layout:tt)*) => {
        UniChunked<Tensor![$(($n * 14))? $real; $(& $($l)? $($mut)?)? @ $($layout)*], U14>
    };
    ($(($n:expr))? $real:ident; $(& $($l:lifetime)? $($mut:ident)?)? @ 15 $($layout:tt)*) => {
        UniChunked<Tensor![$(($n * 15))? $real; $(& $($l)? $($mut)?)? @ $($layout)*], U15>
    };
    ($(($n:expr))? $real:ident; $(& $($l:lifetime)? $($mut:ident)?)? @ 16 $($layout:tt)*) => {
        UniChunked<Tensor![$(($n * 16))? $real; $(& $($l)? $($mut)?)? @ $($layout)*], U16>
    };
    ($(($n:expr))? $real:ident; $(& $($l:lifetime)? $($mut:ident)?)? $($layout:tt)*) => {
        Tensor![(1) $real; $(& $($l)? $($mut)?)? @ $($layout)*]
    };
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

        let ssss33 = <Tensor![f64; S S S S 3 3]>::from_size(&[64, 32, 12, 16, 3, 3]);
        let ssss33view: Tensor![f64; & S S S S 3 3] = ssss33.view();
        assert_eq!(
            Sparse::from_dim(
                &[][..],
                64,
                Chunked::from_offsets(
                    &[0][..],
                    Sparse::from_dim(
                        &[][..],
                        32,
                        Chunked::from_offsets(
                            &[0][..],
                            Sparse::from_dim(
                                &[][..],
                                12,
                                Chunked::from_offsets(
                                    &[0][..],
                                    Sparse::from_dim(
                                        &[][..],
                                        16,
                                        Chunked3::from_flat(Chunked3::from_flat(&[][..]))
                                    )
                                )
                            )
                        )
                    )
                )
            ),
            ssss33view
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
}
