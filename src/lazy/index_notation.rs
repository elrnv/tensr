#![macro_use]

use crate::lazy::*;
use flatk::*;

// The goal is to achieve this type of notation (from taco):
//  let A: Tensor<f64> = /* Constructor */;
//  IndexVar i, j, k;
//  A(i,j) = B(i,j,k) * c(k);

pub trait IExpr<'a, I> {
    type Output;
    fn iexpr(&'a self, index: I) -> Self::Output;
}

impl<'a, I, T: 'a + Clone> IExpr<'a, I> for [T] {
    type Output = SliceIterExpr<'a, T, I>;
    #[inline]
    fn iexpr(&'a self, index: I) -> Self::Output {
        SliceIterExpr {
            _index: index,
            iter: self.iter(),
        }
    }
}

impl<'a, I, T: 'a + Clone> IExpr<'a, I> for Vec<T> {
    type Output = SliceIterExpr<'a, T, I>;
    #[inline]
    fn iexpr(&'a self, index: I) -> Self::Output {
        SliceIterExpr {
            _index: index,
            iter: self.iter(),
        }
    }
}

impl<'a, I, S: View<'a>, N: Copy> IExpr<'a, I> for UniChunked<S, N> {
    type Output = UniChunkedIterExpr<S::Type, N, I>;
    #[inline]
    fn iexpr(&'a self, index: I) -> Self::Output {
        UniChunkedIterExpr {
            index,
            data: self.data.view(),
            chunk_size: self.chunk_size,
        }
    }
}

//impl<'a, I, J, T, R: DenseExpr> Mul<R> for SliceIterExpr<'a, T, (I, J)> {
//    type Output = CwiseMulExpr<Self, R>;
//    #[inline]
//    fn mu(self, rhs: R) -> Self::Output {
//        CwiseMulExpr::new(self, rhs)
//    }
//}

// TODO: Implement this
//macro_rules! expr {
//    () => {};
//}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lazy::constructors::FromShape;

    #[test]
    fn multiply() {
        struct I;
        struct J;
        struct K;
        let _ = <Tensor![f64; D]>::from_shape(&[42]);
        let b = <Tensor![f64; D D 3]>::from_shape(&[64, 42, 3]);
        let _ = b.iexpr((I, J, K));
        //let a: Tensor![f64; D 3] = (b.iexpr::<I, J, K>() * c.iexpr::<J>()).eval::<I, K>();
    }
}
