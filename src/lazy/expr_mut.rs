/**
 * Assignable expressions.
 */
use super::*;

#[derive(Debug)]
pub struct SliceIterExprMut<'a, T, I = ()> {
    // Not yet used (intended for index notation)
    _index: I,
    iter: std::slice::IterMut<'a, T>,
}

impl<'a, T, I> DenseExpr for SliceIterExprMut<'a, T, I> {}

impl<'a, T, Out, I> Iterator for SliceIterExprMut<'a, T, I>
where
    &'a mut T: IntoExpr<Expr = Out>,
{
    type Item = Out;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|x| x.into_expr())
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}
impl<'a, T, I> Expression for SliceIterExprMut<'a, T, I> {}
impl<'a, T, I> ExactSizeIterator for SliceIterExprMut<'a, T, I> where &'a mut T: IntoExpr {}
impl<'a, T, I> ExprSize for SliceIterExprMut<'a, T, I> {
    fn expr_size(&self) -> usize {
        self.iter.size_hint().1.unwrap_or(self.iter.size_hint().0)
    }
}
impl<'a, T, I> TotalExprSize for SliceIterExprMut<'a, T, I> {
    #[inline]
    fn total_size_hint(&self, _cwise_reduce: u32) -> Option<usize> {
        Some(self.iter.size_hint().1.unwrap_or(self.iter.size_hint().0))
    }
}

pub trait ExprMut<'a> {
    type Output;
    fn expr_mut(&'a mut self) -> Self::Output;
}

impl<'a, T: 'a> ExprMut<'a> for [T] {
    type Output = SliceIterExprMut<'a, T, ()>;
    #[inline]
    fn expr_mut(&'a mut self) -> Self::Output {
        SliceIterExprMut {
            _index: (),
            iter: self.iter_mut(),
        }
    }
}

impl<'a, T: 'a> ExprMut<'a> for Vec<T> {
    type Output = SliceIterExprMut<'a, T, ()>;
    #[inline]
    fn expr_mut(&'a mut self) -> Self::Output {
        SliceIterExprMut {
            _index: (),
            iter: self.iter_mut(),
        }
    }
}

impl<'a, S: ViewMut<'a>, N: Copy> ExprMut<'a> for UniChunked<S, N> {
    type Output = UniChunkedIterExpr<S::Type, N>;
    #[inline]
    fn expr_mut(&'a mut self) -> Self::Output {
        UniChunkedIterExpr {
            index: (),
            data: self.data.view_mut(),
            chunk_size: self.chunk_size,
        }
    }
}

impl<'a, S, O> ExprMut<'a> for Chunked<S, O>
where
    S: ViewMut<'a>,
    O: View<'a, Type = Offsets<&'a [usize]>>,
{
    type Output = ChunkedIterExpr<'a, S::Type>;
    #[inline]
    fn expr_mut(&'a mut self) -> Self::Output {
        ChunkedIterExpr {
            data: self.data.view_mut(),
            chunk_sizes: self.chunks.view().into_sizes(),
        }
    }
}

impl<'a, S, T, I> ExprMut<'a> for Sparse<S, T, I>
where
    S: ViewMut<'a>,
    T: View<'a>,
    I: View<'a, Type = &'a [usize]>,
    SparseIterExpr<'a, S::Type, T::Type>: Iterator,
{
    type Output = SparseExpr<SparseIterExpr<'a, S::Type, T::Type>>;
    #[inline]
    fn expr_mut(&'a mut self) -> Self::Output {
        SparseExpr::new(SparseIterExpr {
            indices: self.selection.indices.view(),
            source: self.source.view_mut(),
            target: self.selection.target.view(),
        })
    }
}

impl<'a, S, I> ExprMut<'a> for Subset<S, I>
where
    S: ViewMut<'a>,
    I: AsRef<[usize]>,
{
    type Output = SubsetIterExpr<'a, S::Type>;
    #[inline]
    fn expr_mut(&'a mut self) -> Self::Output {
        let (indices, data) = self.view_mut().into_raw();
        SubsetIterExpr { indices, data }
    }
}

impl<'a, T: ?Sized, D: 'a + ?Sized> ExprMut<'a> for Tensor<T>
where
    Self: AsMutData<Data = D>,
    D: ExprMut<'a>,
{
    type Output = D::Output;
    #[inline]
    fn expr_mut(&'a mut self) -> Self::Output {
        self.as_mut_data().expr_mut()
    }
}
