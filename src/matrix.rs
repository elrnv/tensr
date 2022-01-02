//!
//! Common matrix types and operations.
//!

mod sprs_compat;
mod ssblock;

use std::ops::{Add, Mul, MulAssign};

use num_traits::{Float, Zero};
use rayon::prelude::*;

use super::*;

pub use sprs_compat::*;
pub use ssblock::*;

type Dim = std::ops::RangeTo<usize>;

pub trait SparseMatrix {
    fn num_non_zeros(&self) -> usize;
}

pub trait SparseBlockMatrix {
    fn num_non_zero_blocks(&self) -> usize;
}

/// This trait defines information provided by any matrx type.
pub trait Matrix {
    type Transpose;
    fn transpose(self) -> Self::Transpose;
    fn num_rows(&self) -> usize;
    fn num_cols(&self) -> usize;
}

/// A block matrix is a matrix of of smaller matrices organized in blocks. It can also be
/// interpreted as a fourth order tensor.
pub trait BlockMatrix {
    fn num_rows_per_block(&self) -> usize;
    fn num_cols_per_block(&self) -> usize;
    fn num_total_rows(&self) -> usize;
    fn num_total_cols(&self) -> usize;
}

/*
 * One-dimentional vectors
 */

impl<T: Scalar> Matrix for Tensor<Vec<T>> {
    type Transpose = Transpose<Self>;
    fn transpose(self) -> Self::Transpose {
        Transpose(self)
    }
    fn num_cols(&self) -> usize {
        1
    }
    fn num_rows(&self) -> usize {
        self.data.len()
    }
}

impl<T: Scalar> Matrix for &Tensor<[T]> {
    type Transpose = Transpose<Self>;
    fn transpose(self) -> Self::Transpose {
        Transpose(self)
    }
    fn num_cols(&self) -> usize {
        1
    }
    fn num_rows(&self) -> usize {
        self.data.len()
    }
}

impl<T: Scalar> Matrix for &mut Tensor<[T]> {
    type Transpose = Transpose<Self>;
    fn transpose(self) -> Self::Transpose {
        Transpose(self)
    }
    fn num_cols(&self) -> usize {
        1
    }
    fn num_rows(&self) -> usize {
        self.data.len()
    }
}

/*
 * Matrices
 */

/// Row-major dense matrix with dynamic number of rows and N columns, where N can be `usize` or a
/// constant.
pub type DMatrixBase<T, N = usize> = Tensor<UniChunked<T, N>>;
pub type DMatrix<T = f64, N = usize> = DMatrixBase<Tensor<Vec<T>>, N>;
pub type DMatrixView<'a, T = f64, N = usize> = DMatrixBase<&'a Tensor<[T]>, N>;

impl<N, T> Matrix for DMatrixBase<T, N>
where
    N: Dimension,
    Self: Set,
{
    type Transpose = Transpose<Self>;
    fn transpose(self) -> Self::Transpose {
        Transpose(self)
    }
    fn num_cols(&self) -> usize {
        self.chunk_size()
    }
    fn num_rows(&self) -> usize {
        self.len()
    }
}

/// Row-major dense matrix of row-major NxM blocks where N is the number of rows an M number of
/// columns.
pub type DBlockMatrixBase<T, N, M> = DMatrixBase<Tensor<UniChunked<Tensor<UniChunked<T, M>>, N>>>;
pub type DBlockMatrix<T = f64, N = usize, M = usize> = DBlockMatrixBase<Tensor<Vec<T>>, N, M>;
pub type DBlockMatrixView<'a, T = f64, N = usize, M = usize> =
    DBlockMatrixBase<&'a Tensor<[T]>, N, M>;

/// Row-major dense matrix of row-major 3x3 blocks.
pub type DBlockMatrix3<T = f64> = DBlockMatrix<T, U3, U3>;
pub type DBlockMatrix3View<'a, T = f64> = DBlockMatrixView<'a, T, U3, U3>;

/// Dense-row sparse-column row-major matrix. AKA CSR matrix.
pub type DSMatrixBase<T, I> = Tensor<Chunked<Tensor<Sparse<T, Dim, I>>, Offsets<I>>>;
pub type DSMatrix<T = f64, I = Vec<usize>> = DSMatrixBase<Tensor<Vec<T>>, I>;
pub type DSMatrixView<'a, T = f64> = DSMatrixBase<&'a Tensor<[T]>, &'a [usize]>;

impl<S: IntoData, I: AsIndexSlice> Matrix for DSMatrixBase<S, I>
where
    S::Data: Set,
{
    type Transpose = Transpose<Self>;
    fn transpose(self) -> Self::Transpose {
        Transpose(self)
    }
    fn num_cols(&self) -> usize {
        self.as_data().data().selection().target.distance()
    }
    fn num_rows(&self) -> usize {
        self.as_data().len()
    }
}

impl<S, I> SparseMatrix for DSMatrixBase<S, I>
where
    S: Storage,
    S::Storage: Set,
{
    fn num_non_zeros(&self) -> usize {
        self.storage().len()
    }
}

impl<T: Scalar> DSMatrix<T> {
    /// Construct a sparse matrix from a given iterator of triplets.
    pub fn from_triplets_iter<I>(iter: I, num_rows: usize, num_cols: usize) -> Self
    where
        I: Iterator<Item = (usize, usize, T)>,
    {
        Self::from_triplets_iter_uncompressed(iter, num_rows, num_cols).compressed()
    }

    /// Construct a possibly uncompressed sparse matrix from a given iterator of triplets.
    ///
    /// This is useful if the caller needs to prune the matrix anyways, which will compress it in
    /// the process, thus saving an extra pass through the values.
    pub fn from_triplets_iter_uncompressed<I>(iter: I, num_rows: usize, num_cols: usize) -> Self
    where
        I: Iterator<Item = (usize, usize, T)>,
    {
        let mut triplets: Vec<_> = iter.collect();
        triplets.sort_by_key(|&(row, _, _)| row);
        Self::from_sorted_triplets_iter_uncompressed(triplets.into_iter(), num_rows, num_cols)
    }

    /// Assume that rows are monotonically increasing in the iterator. Columns don't have an order
    /// restriction.
    pub fn from_sorted_triplets_iter<I>(iter: I, num_rows: usize, num_cols: usize) -> Self
    where
        I: Iterator<Item = (usize, usize, T)>,
    {
        Self::from_sorted_triplets_iter_uncompressed(iter, num_rows, num_cols).compressed()
    }

    /// Assume that rows are monotonically increasing in the iterator. Columns don't have an order
    /// restriction.
    pub fn from_sorted_triplets_iter_uncompressed<I>(
        iter: I,
        num_rows: usize,
        num_cols: usize,
    ) -> Self
    where
        I: Iterator<Item = (usize, usize, T)>,
    {
        let cap = iter.size_hint().0;
        let mut cols = Vec::with_capacity(cap);
        let mut vals: Vec<T> = Vec::with_capacity(cap);
        let mut offsets = Vec::with_capacity(num_rows);

        let mut prev_row = 0; // offset by +1 so we don't have to convert between isize.
        for (row, col, val) in iter {
            assert!(row + 1 >= prev_row); // We assume that rows are monotonically increasing.

            if row + 1 != prev_row {
                prev_row = row + 1;
                offsets.push(cols.len());
            }

            cols.push(col);
            vals.push(val);
        }
        offsets.push(cols.len());
        offsets.shrink_to_fit();

        let mut col_data = Chunked::from_offsets(offsets, Sparse::from_dim(cols, num_cols, vals));

        col_data.sort_chunks_by_index();

        col_data.into_tensor()
    }
}

//impl<S, I> DSMatrixBase<S, I>
//where
//    Self: for<'a> View<'a, Type = DSMatrixView<'a>>,
//{
//    /// Compress the matrix representation by consolidating duplicate entries.
//    pub fn compressed(&self) -> DSMatrix {
//        self.view()
//            .into_data()
//            .compressed(|a, &b| *a += b)
//            .into_tensor()
//    }
//}
//
//impl<S, I> DSMatrixBase<S, I>
//where
//    Self: for<'a> View<'a, Type = DSMatrixView<'a>>,
//{
//    /// Remove all elements that do not satisfy the given predicate and compress the resulting matrix.
//    pub fn pruned(&self, keep: impl Fn(usize, usize, &f64) -> bool) -> DSMatrix {
//        self.view()
//            .into_data()
//            .pruned(|a, &b| *a += b, keep)
//            .into_tensor()
//    }
//}

impl<T: Scalar, I: AsIndexSlice> DSMatrix<T, I> {
    /// Compress the matrix representation by consolidating duplicate entries.
    pub fn compressed(&self) -> DSMatrix<T> {
        self.view()
            .into_data()
            .compressed(|a, &b| *a += b)
            .into_tensor()
    }
}

impl<T: Scalar, I: AsIndexSlice> DSMatrix<T, I> {
    /// Remove all elements that do not satisfy the given predicate and compress
    /// the resulting matrix.
    ///
    /// The `mapping` function allows the caller to keep track of how the global
    /// index array changes as a result. For each kept element in the original
    /// matrix, `mapping` will be called with the first parameter being the
    /// original index and second parameter being the destination index in the
    /// output matrix.
    pub fn pruned(
        &self,
        keep: impl Fn(usize, usize, &T) -> bool,
        mapping: impl FnMut(usize, usize),
    ) -> DSMatrix<T> {
        self.view()
            .into_data()
            .pruned(|a, &b| *a += b, keep, mapping)
            .into_tensor()
    }
}

/*
 * A diagonal matrix has the same structure as a vector, so it needs a newtype to distinguish it
 * from such.
 */

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct DiagonalMatrixBase<T, I = Box<[usize]>>(Subset<T, I>);
pub type DiagonalMatrix<T = f64, I = Box<[usize]>> = DiagonalMatrixBase<Vec<T>, I>;
pub type DiagonalMatrixView<'a, T = f64> = DiagonalMatrixBase<&'a [T], &'a [usize]>;
pub type DiagonalMatrixViewMut<'a, T = f64> = DiagonalMatrixBase<&'a mut [T], &'a [usize]>;

impl<S: Set> DiagonalMatrixBase<S, Box<[usize]>> {
    /// A generic constructor that transforms the input into the underlying storage type. This
    /// sometimes requires additional generic parameters to be explicitly specified.
    /// This function assumes `Box<[usize]>` as a placeholder for indices where the subset is
    /// entire.
    pub fn new(set: S) -> Self {
        DiagonalMatrixBase(Subset::all(set))
    }
}

impl<S: Set, I: AsRef<[usize]>> DiagonalMatrixBase<S, I> {
    /// Explicit constructor from subsets.
    pub fn from_subset(subset: Subset<S, I>) -> Self {
        DiagonalMatrixBase(subset.into())
    }
    /// Produce a mutable view of this diagonal block matrix as a tensor. When interprepted as a tensor the data
    /// contained in this matrix represents a dense matrix with `self.0.len()` rows and `N` columns.
    pub fn tensor_view_mut<'a, T>(&'a mut self) -> Tensor<SubsetView<'a, T>>
    where
        S: ViewMut<'a>,
        T: IntoData<Data = S::Type>,
        S::Type: IntoTensor<Tensor = T>,
    {
        self.0.view_mut().into_tensor()
    }

    /// Produce an immutable view of this diagonal block matrix as a tensor. When interprepted as a tensor the data
    /// contained in this matrix represents a dense matrix with `self.0.len()` rows and `N` columns.
    pub fn tensor_view<'a, T>(&'a self) -> Tensor<SubsetView<'a, T>>
    where
        S: View<'a>,
        T: IntoData<Data = S::Type>,
        S::Type: IntoTensor<Tensor = T>,
    {
        self.0.view().into_tensor()
    }
}

impl<S: Set> Matrix for DiagonalMatrixBase<S> {
    type Transpose = Self;
    fn transpose(self) -> Self {
        self
    }
    fn num_cols(&self) -> usize {
        self.0.len()
    }
    fn num_rows(&self) -> usize {
        self.0.len()
    }
}

impl<T, S, I> Norm<T> for DiagonalMatrixBase<S, I>
where
    T: Scalar,
    Subset<S, I>: for<'a> ViewIterator<'a, Item = &'a T>,
    T: num_traits::FromPrimitive,
{
    fn lp_norm(&self, norm: LpNorm) -> T
    where
        T: Float,
    {
        match norm {
            LpNorm::P(p) => {
                self.0.view_iter().map(|x| x.abs().powi(p)).sum::<T>().powf(
                    T::one() / T::from_i32(p).expect("Failed to convert integer to flaot type."),
                )
            }
            LpNorm::Inf => self
                .0
                .view_iter()
                .map(|x| x.abs())
                .max_by(|x, y| {
                    x.partial_cmp(y)
                        .expect("Detected NaN when computing Inf-norm.")
                })
                .unwrap_or(T::zero()),
        }
    }
    fn norm_squared(&self) -> T {
        self.0.view_iter().map(|&x| x * x).sum::<T>()
    }
}

impl<S: Set> SparseMatrix for DiagonalMatrixBase<S> {
    fn num_non_zeros(&self) -> usize {
        self.num_rows()
    }
}

impl<S: Viewed, I> Viewed for DiagonalMatrixBase<S, I> {}

impl<'a, S: Set + View<'a, Type = &'a [f64]>, I: AsRef<[usize]>> View<'a>
    for DiagonalMatrixBase<S, I>
{
    type Type = DiagonalMatrixView<'a>;
    fn view(&'a self) -> Self::Type {
        DiagonalMatrixBase(View::view(&self.0))
    }
}

impl<'a, S: Set + ViewMut<'a, Type = &'a mut [f64]>, I: AsRef<[usize]>> ViewMut<'a>
    for DiagonalMatrixBase<S, I>
{
    type Type = DiagonalMatrixViewMut<'a>;
    fn view_mut(&'a mut self) -> Self::Type {
        DiagonalMatrixBase(ViewMut::view_mut(&mut self.0))
    }
}

/// A diagonal matrix of `N` sized chunks. this is not to be confused with block diagonal matrix,
/// which may contain off-diagonal elements in each block. This is a purely diagonal matrix, whose
/// diagonal elements are grouped into `N` sized chunks.
//
// TODO: Unify specialized matrix types like DiagonalBlockMatrixBase to have a similar api to
// Tensors.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct DiagonalBlockMatrixBase<S, I = Box<[usize]>, N = usize>(pub Subset<UniChunked<S, N>, I>);
pub type DiagonalBlockMatrix<T = f64, I = Box<[usize]>, N = usize> =
    DiagonalBlockMatrixBase<Vec<T>, I, N>;
pub type DiagonalBlockMatrixView<'a, T = f64, N = usize> =
    DiagonalBlockMatrixBase<&'a [T], &'a [usize], N>;
pub type DiagonalBlockMatrixViewMut<'a, T = f64, N = usize> =
    DiagonalBlockMatrixBase<&'a mut [T], &'a [usize], N>;

pub type DiagonalBlockMatrix3<T = f64, I = Box<[usize]>> = DiagonalBlockMatrix<T, I, U3>;
pub type DiagonalBlockMatrix3View<'a, T = f64> = DiagonalBlockMatrixView<'a, T, U3>;

impl<S, N: Dimension> DiagonalBlockMatrixBase<S, Box<[usize]>, N>
where
    UniChunked<S, N>: Set,
{
    /// A generic constructor that transforms the input into the underlying storage type. This
    /// sometimes requires additional generic parameters to be explicitly specified.
    /// This function assumes `Box<[usize]>` as a placeholder for indices where the subset is
    /// entire.
    pub fn new(chunks: UniChunked<S, N>) -> Self {
        DiagonalBlockMatrixBase(Subset::all(chunks))
    }
}

impl<'a, S, N: Dimension> DiagonalBlockMatrixBase<S, &'a [usize], N>
where
    UniChunked<S, N>: Set,
{
    pub fn view(chunks: UniChunked<S, N>) -> Self {
        DiagonalBlockMatrixBase(Subset::all(chunks))
    }
}

impl<S, I: AsRef<[usize]>, N: Dimension> DiagonalBlockMatrixBase<S, I, N>
where
    UniChunked<S, N>: Set,
{
    /// Explicit constructor from subsets.
    pub fn from_subset(chunks: Subset<UniChunked<S, N>, I>) -> Self {
        DiagonalBlockMatrixBase(chunks)
    }
    /// Produce a mutable view of this diagonal block matrix as a tensor. When interprepted as a tensor the data
    /// contained in this matrix represents a dense matrix with `self.0.len()` rows and `N` columns.
    pub fn tensor_view_mut<'a, T>(&'a mut self) -> Tensor<SubsetView<'a, Tensor<UniChunked<T, N>>>>
    where
        S: Set + ViewMut<'a>,
        T: IntoData<Data = S::Type>,
        S::Type: IntoTensor<Tensor = T>,
    {
        ViewMut::view_mut(&mut self.0).into_tensor()
    }

    /// Produce an immutable view of this diagonal block matrix as a tensor. When interprepted as a tensor the data
    /// contained in this matrix represents a dense matrix with `self.0.len()` rows and `N` columns.
    pub fn tensor_view<'a, T>(&'a self) -> Tensor<SubsetView<'a, Tensor<UniChunked<T, N>>>>
    where
        S: Set + View<'a>,
        T: IntoData<Data = S::Type>,
        S::Type: IntoTensor<Tensor = T>,
    {
        self.0.view().into_tensor()
    }
}

impl<S, N: Dimension> DiagonalBlockMatrixBase<S, Box<[usize]>, N>
where
    UniChunked<S, N>: Set,
{
    /// Explicit constructor from uniformly chunked collections.
    pub fn from_uniform(chunks: UniChunked<S, N>) -> Self {
        DiagonalBlockMatrixBase(Subset::all(chunks))
    }
}
impl<S, N> DiagonalBlockMatrixBase<S, Box<[usize]>, U<N>>
where
    UniChunked<S, U<N>>: Set,
    N: Unsigned + Default,
    S: Set,
{
    pub fn from_flat(chunks: S) -> Self {
        DiagonalBlockMatrixBase(Subset::all(UniChunked::from_flat(chunks)))
    }
}

impl<S, I: AsRef<[usize]>, N: Dimension> BlockMatrix for DiagonalBlockMatrixBase<S, I, N>
where
    UniChunked<S, N>: Set,
{
    fn num_cols_per_block(&self) -> usize {
        self.0.chunk_size()
    }
    fn num_rows_per_block(&self) -> usize {
        self.0.chunk_size()
    }
    fn num_total_cols(&self) -> usize {
        self.num_cols() * self.num_cols_per_block()
    }
    fn num_total_rows(&self) -> usize {
        self.num_rows() * self.num_rows_per_block()
    }
}

impl<S, I: AsRef<[usize]>, N: Dimension> Matrix for DiagonalBlockMatrixBase<S, I, N>
where
    UniChunked<S, N>: Set,
{
    type Transpose = Self;
    fn transpose(self) -> Self {
        self
    }
    fn num_cols(&self) -> usize {
        self.0.len()
    }
    fn num_rows(&self) -> usize {
        self.0.len()
    }
}

impl<T, I> Norm<T> for DiagonalBlockMatrix3<T, I>
where
    T: Scalar,
    Subset<Chunked3<Vec<T>>, I>: for<'a> ViewIterator<'a, Item = &'a [T; 3]>,
    T: num_traits::FromPrimitive,
{
    fn lp_norm(&self, norm: LpNorm) -> T
    where
        T: Float,
    {
        match norm {
            LpNorm::P(p) => self
                .0
                .view_iter()
                .map(|v| v.as_tensor().map(|x| x.abs().powi(p)).sum())
                .sum::<T>()
                .powf(T::one() / T::from_i32(p).expect("Failed to convert integer to flaot type.")),
            LpNorm::Inf => self
                .0
                .view_iter()
                .flat_map(|v| v.iter().map(|x| x.abs()))
                .max_by(|x, y| {
                    x.partial_cmp(y)
                        .expect("Detected NaN when computing Inf-norm.")
                })
                .unwrap_or(T::zero()),
        }
    }
    fn norm_squared(&self) -> T {
        self.0
            .view_iter()
            .map(|&x| x.as_tensor().norm_squared())
            .sum::<T>()
    }
}

impl<S, I: AsRef<[usize]>, N: Dimension> SparseMatrix for DiagonalBlockMatrixBase<S, I, N>
where
    UniChunked<S, N>: Set,
{
    fn num_non_zeros(&self) -> usize {
        self.num_total_rows()
    }
}

impl<S, I: AsRef<[usize]>, N: Dimension> SparseBlockMatrix for DiagonalBlockMatrixBase<S, I, N>
where
    UniChunked<S, N>: Set,
{
    fn num_non_zero_blocks(&self) -> usize {
        self.num_rows()
    }
}

impl<S: Viewed, I, N> Viewed for DiagonalBlockMatrixBase<S, I, N> {}

impl<'a, T: 'a, S: Set + View<'a, Type = &'a [T]>, I: AsRef<[usize]>, N: Copy> View<'a>
    for DiagonalBlockMatrixBase<S, I, N>
where
    UniChunked<S, N>: Set,
{
    type Type = DiagonalBlockMatrixView<'a, T, N>;
    fn view(&'a self) -> Self::Type {
        DiagonalBlockMatrixBase(View::view(&self.0))
    }
}

impl<'a, T: 'a, S: Set + ViewMut<'a, Type = &'a mut [T]>, I: AsRef<[usize]>, N: Copy> ViewMut<'a>
    for DiagonalBlockMatrixBase<S, I, N>
where
    UniChunked<S, N>: Set,
{
    type Type = DiagonalBlockMatrixViewMut<'a, T, N>;
    fn view_mut(&'a mut self) -> Self::Type {
        DiagonalBlockMatrixBase(ViewMut::view_mut(&mut self.0))
    }
}

/*
 * Block Diagonal matrices
 */

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BlockDiagonalMatrixBase<S, I = Box<[usize]>, N = usize, M = usize>(
    pub Subset<UniChunked<UniChunked<S, M>, N>, I>,
);
pub type BlockDiagonalMatrix<T = f64, I = Box<[usize]>, N = usize, M = usize> =
    BlockDiagonalMatrixBase<Vec<T>, I, N, M>;
pub type BlockDiagonalMatrixView<'a, T = f64, N = usize, M = usize> =
    BlockDiagonalMatrixBase<&'a [T], &'a [usize], N, M>;
pub type BlockDiagonalMatrixViewMut<'a, T = f64, N = usize, M = usize> =
    BlockDiagonalMatrixBase<&'a mut [T], &'a [usize], N, M>;

pub type BlockDiagonalMatrix3x2<T = f64, I = Box<[usize]>> = BlockDiagonalMatrix<T, I, U3, U2>;
pub type BlockDiagonalMatrix3x2View<'a, T = f64> = BlockDiagonalMatrixView<'a, T, U3, U2>;

pub type BlockDiagonalMatrix3x1<T = f64, I = Box<[usize]>> = BlockDiagonalMatrix<T, I, U3, U1>;
pub type BlockDiagonalMatrix3x1View<'a, T = f64> = BlockDiagonalMatrixView<'a, T, U3, U1>;

pub type BlockDiagonalMatrix2<T = f64, I = Box<[usize]>> = BlockDiagonalMatrix<T, I, U2, U2>;
pub type BlockDiagonalMatrix2View<'a, T = f64> = BlockDiagonalMatrixView<'a, T, U2, U2>;

pub type BlockDiagonalMatrix3<T = f64, I = Box<[usize]>> = BlockDiagonalMatrix<T, I, U3, U3>;
pub type BlockDiagonalMatrix3View<'a, T = f64> = BlockDiagonalMatrixView<'a, T, U3, U3>;

impl<S, N: Dimension, M: Dimension> BlockDiagonalMatrixBase<S, Box<[usize]>, N, M>
where
    UniChunked<UniChunked<S, M>, N>: Set,
    UniChunked<S, M>: Set,
{
    /// A generic constructor that transforms the input into the underlying storage type. This
    /// sometimes requires additional generic parameters to be explicitly specified.
    /// This function assumes `Box<[usize]>` as a placeholder for indices where the subset is
    /// entire.
    pub fn new(chunks: UniChunked<UniChunked<S, M>, N>) -> Self {
        BlockDiagonalMatrixBase(Subset::all(chunks))
    }
}

impl BlockDiagonalMatrix3x1 {
    pub fn negate(&mut self) {
        for mut x in self.0.iter_mut() {
            for x in x.iter_mut() {
                for x in x.iter_mut() {
                    *x = -*x;
                }
            }
        }
    }
}

impl<S, I: AsRef<[usize]>, N: Dimension, M: Dimension> BlockDiagonalMatrixBase<S, I, N, M>
where
    UniChunked<UniChunked<S, M>, N>: Set,
    UniChunked<S, M>: Set,
{
    /// Explicit constructor from subsets.
    pub fn from_subset(chunks: Subset<UniChunked<UniChunked<S, M>, N>, I>) -> Self {
        BlockDiagonalMatrixBase(chunks.into())
    }
}

impl<S, N: Dimension, M: Dimension> BlockDiagonalMatrixBase<S, Box<[usize]>, N, M>
where
    UniChunked<UniChunked<S, M>, N>: Set,
    UniChunked<S, M>: Set,
{
    /// Explicit constructor from uniformly chunked collections.
    pub fn from_uniform(chunks: UniChunked<UniChunked<S, M>, N>) -> Self {
        BlockDiagonalMatrixBase(Subset::all(chunks))
    }
}
impl<S, N, M> BlockDiagonalMatrixBase<S, Box<[usize]>, U<N>, U<M>>
where
    UniChunked<UniChunked<S, U<M>>, U<N>>: Set,
    UniChunked<S, U<M>>: Set,
    N: Unsigned + Default,
    M: Unsigned + Default,
    S: Set,
{
    pub fn from_flat(chunks: S) -> Self {
        BlockDiagonalMatrixBase(Subset::all(UniChunked::from_flat(UniChunked::from_flat(
            chunks,
        ))))
    }
}

impl<S, I, N: Dimension, M: Dimension> BlockMatrix for BlockDiagonalMatrixBase<S, I, N, M>
where
    UniChunked<UniChunked<S, M>, N>: Set,
    UniChunked<S, M>: Set,
    I: AsRef<[usize]>,
{
    fn num_cols_per_block(&self) -> usize {
        self.0.chunk_size()
    }
    fn num_rows_per_block(&self) -> usize {
        self.0.chunk_size()
    }
    fn num_total_cols(&self) -> usize {
        self.num_cols() * self.num_cols_per_block()
    }
    fn num_total_rows(&self) -> usize {
        self.num_rows() * self.num_rows_per_block()
    }
}

impl<S, I, N: Dimension, M: Dimension> Matrix for BlockDiagonalMatrixBase<S, I, N, M>
where
    UniChunked<UniChunked<S, M>, N>: Set,
    UniChunked<S, M>: Set,
    I: AsRef<[usize]>,
{
    type Transpose = Transpose<Self>;
    fn transpose(self) -> Self::Transpose {
        Transpose(self)
    }
    fn num_cols(&self) -> usize {
        self.0.len()
    }
    fn num_rows(&self) -> usize {
        self.0.len()
    }
}

impl<T, I> Norm<T> for BlockDiagonalMatrix3<T, I>
where
    T: Scalar,
    Subset<Chunked3<Chunked3<Vec<T>>>, I>: for<'a> ViewIterator<'a, Item = &'a [[T; 3]; 3]>,
    T: num_traits::FromPrimitive,
{
    fn lp_norm(&self, norm: LpNorm) -> T
    where
        T: Float,
    {
        match norm {
            LpNorm::P(p) => self
                .0
                .view_iter()
                .map(|v| v.as_tensor().map_inner(|x| x.abs().powi(p)).sum_inner())
                .sum::<T>()
                .powf(T::one() / T::from_i32(p).expect("Failed to convert integer to flaot type.")),
            LpNorm::Inf => self
                .0
                .view_iter()
                .flat_map(|v| v.iter().flat_map(|v| v.iter()))
                .max_by(|x, y| {
                    x.partial_cmp(y)
                        .expect("Detected NaN when computing Inf-norm.")
                })
                .cloned()
                .unwrap_or(T::zero()),
        }
    }
    fn norm_squared(&self) -> T {
        self.0
            .view_iter()
            .map(|&x| x.as_tensor().frob_norm_squared())
            .sum::<T>()
    }
}

impl<S, I, N: Dimension, M: Dimension> SparseMatrix for BlockDiagonalMatrixBase<S, I, N, M>
where
    UniChunked<UniChunked<S, M>, N>: Set,
    UniChunked<S, M>: Set,
    I: AsRef<[usize]>,
{
    fn num_non_zeros(&self) -> usize {
        self.num_total_rows()
    }
}

impl<S, I, N: Dimension, M: Dimension> SparseBlockMatrix for BlockDiagonalMatrixBase<S, I, N, M>
where
    UniChunked<UniChunked<S, M>, N>: Set,
    UniChunked<S, M>: Set,
    I: AsRef<[usize]>,
{
    fn num_non_zero_blocks(&self) -> usize {
        self.num_rows()
    }
}

impl<S: Viewed, I, N, M> Viewed for BlockDiagonalMatrixBase<S, I, N, M> {}

impl<'a, S: Set + View<'a, Type = &'a [f64]>, I: AsRef<[usize]>, N: Copy, M: Copy> View<'a>
    for BlockDiagonalMatrixBase<S, I, N, M>
where
    UniChunked<UniChunked<S, M>, N>: Set,
    UniChunked<S, M>: Set,
{
    type Type = BlockDiagonalMatrixView<'a, f64, N, M>;
    fn view(&'a self) -> Self::Type {
        BlockDiagonalMatrixBase(View::view(&self.0))
    }
}

impl<'a, S: Set + ViewMut<'a, Type = &'a mut [f64]>, I: AsRef<[usize]>, N: Copy, M: Copy>
    ViewMut<'a> for BlockDiagonalMatrixBase<S, I, N, M>
where
    UniChunked<UniChunked<S, M>, N>: Set,
    UniChunked<S, M>: Set,
{
    type Type = BlockDiagonalMatrixViewMut<'a, f64, N, M>;
    fn view_mut(&'a mut self) -> Self::Type {
        BlockDiagonalMatrixBase(ViewMut::view_mut(&mut self.0))
    }
}

// TODO: make this generic over the number
impl<I: AsRef<[usize]>, J: AsRef<[usize]>> Mul<BlockDiagonalMatrix3x2<f64, J>>
    for Transpose<BlockDiagonalMatrix3x2<f64, I>>
{
    type Output = BlockDiagonalMatrix2;
    fn mul(self, other: BlockDiagonalMatrix3x2<f64, J>) -> Self::Output {
        let ref self_data = (self.0).0;
        let ref other_data = other.0;
        assert_eq!(Set::len(self_data), other_data.len());
        assert_eq!(2, self_data.inner_chunk_size());
        let mut out = BlockDiagonalMatrix::from_flat(vec![0.0; 4 * self_data.len()]);
        for ((out_block, lhs_block), rhs_block) in out
            .0
            .iter_mut()
            .zip(self_data.iter())
            .zip(other_data.iter())
        {
            let out_mtx: &mut Matrix2<f64> = out_block.as_matrix();
            *out_mtx = lhs_block.as_matrix().transpose() * *rhs_block.as_matrix();
        }
        out
    }
}

impl<I: AsRef<[usize]>, J: AsRef<[usize]>> Mul<BlockDiagonalMatrix3x1<f64, J>>
    for Transpose<BlockDiagonalMatrix3x1<f64, I>>
{
    type Output = DiagonalMatrix;
    fn mul(self, other: BlockDiagonalMatrix3x1<f64, J>) -> Self::Output {
        let ref self_data = (self.0).0;
        let ref other_data = other.0;
        assert_eq!(Set::len(self_data), other_data.len());
        assert_eq!(self_data.inner_chunk_size(), 1);
        let mut out = DiagonalMatrixBase::new(vec![0.0; self_data.len()]);
        for ((out_entry, lhs_block), rhs_block) in out
            .0
            .iter_mut()
            .zip(self_data.iter())
            .zip(other_data.iter())
        {
            *out_entry = (lhs_block.as_matrix().transpose() * *rhs_block.as_matrix()).data[0][0];
        }
        out
    }
}

/// Dense-row sparse-column row-major 3x3 block matrix. Block version of CSR.
pub type DSBlockMatrixBase<S, I = Vec<usize>, N = usize, M = usize> = Tensor<
    Chunked<Tensor<Sparse<Tensor<UniChunked<Tensor<UniChunked<S, M>>, N>>, Dim, I>>, Offsets<I>>,
>;
pub type DSBlockMatrix<T = f64, I = Vec<usize>, N = usize, M = usize> =
    DSBlockMatrixBase<Tensor<Vec<T>>, I, N, M>;
pub type DSBlockMatrixView<'a, T = f64, N = usize, M = usize> =
    DSBlockMatrixBase<&'a Tensor<[T]>, &'a [usize], N, M>;

pub type DSBlockMatrix2<T = f64, I = Vec<usize>> = DSBlockMatrix<T, I, U2, U2>;
pub type DSBlockMatrix2View<'a, T = f64> = DSBlockMatrixView<'a, T, U2, U2>;

pub type DSBlockMatrix3<T = f64, I = Vec<usize>> = DSBlockMatrix<T, I, U3, U3>;
pub type DSBlockMatrix3View<'a, T = f64> = DSBlockMatrixView<'a, T, U3, U3>;

pub type DSBlockMatrix1x3<T = f64, I = Vec<usize>> = DSBlockMatrix<T, I, U1, U3>;
pub type DSBlockMatrix1x3View<'a, T = f64> = DSBlockMatrixView<'a, T, U1, U3>;

pub type DSBlockMatrix3x1<T = f64, I = Vec<usize>> = DSBlockMatrix<T, I, U3, U1>;
pub type DSBlockMatrix3x1View<'a, T = f64> = DSBlockMatrixView<'a, T, U3, U1>;

impl<S: Set + IntoData, I: Set, N: Dimension, M: Dimension> BlockMatrix
    for DSBlockMatrixBase<S, I, N, M>
where
    Self: Matrix,
{
    fn num_cols_per_block(&self) -> usize {
        self.as_data().data().source().data().chunk_size()
    }
    fn num_rows_per_block(&self) -> usize {
        self.as_data().data().source().chunk_size()
    }
    fn num_total_cols(&self) -> usize {
        self.num_cols() * self.num_cols_per_block()
    }
    fn num_total_rows(&self) -> usize {
        self.num_rows() * self.num_rows_per_block()
    }
}

impl<S: Set + IntoData, I, N, M> SparseBlockMatrix for DSBlockMatrixBase<S, I, N, M>
where
    UniChunked<UniChunked<S::Data, M>, N>: Set,
{
    fn num_non_zero_blocks(&self) -> usize {
        self.as_data().data().source().len()
    }
}

impl<T: Scalar> DSBlockMatrix3<T> {
    /// Assume that rows are monotonically increasing in the iterator. Columns don't have an order
    /// restriction.
    pub fn from_block_triplets_iter<I>(iter: I, num_rows: usize, num_cols: usize) -> Self
    where
        I: Iterator<Item = (usize, usize, [[T; 3]; 3])>,
    {
        let cap = iter.size_hint().0;
        let mut cols = Vec::with_capacity(cap);
        let mut blocks: Vec<[T; 3]> = Vec::with_capacity(cap);
        let mut offsets = Vec::with_capacity(num_rows);

        let mut prev_row = 0; // offset by +1 so we don't have to convert between isize.
        for (row, col, block) in iter {
            assert!(row + 1 >= prev_row); // We assume that rows are monotonically increasing.

            if row + 1 != prev_row {
                prev_row = row + 1;
                offsets.push(cols.len());
            }

            cols.push(col);
            // Push each row at a time to produce a Vec<[f64; 3]>
            blocks.push(block[0]);
            blocks.push(block[1]);
            blocks.push(block[2]);
        }
        offsets.push(cols.len());
        offsets.shrink_to_fit();

        let mut col_data = Chunked::from_offsets(
            offsets,
            Sparse::from_dim(
                cols,
                num_cols,
                Chunked3::from_flat(Chunked3::from_array_vec(blocks)),
            ),
        );

        col_data.sort_chunks_by_index();

        col_data.into_tensor().compressed()
    }
}

impl<T: Scalar, I: AsIndexSlice> DSBlockMatrix3<T, I> {
    /// Compress the matrix representation by consolidating duplicate entries.
    pub fn compressed(&self) -> DSBlockMatrix3<T> {
        self.as_data()
            .view()
            .compressed(|a, b| *a.as_mut_arrays().as_mut_tensor() += b.into_arrays().as_tensor())
            .into_tensor()
    }
}

impl<T: Scalar, I: AsIndexSlice> DSBlockMatrix3<T, I> {
    /// Remove all elements that do not satisfy the given predicate and compress the resulting matrix.
    pub fn pruned(&self, keep: impl Fn(usize, usize, &Matrix3<T>) -> bool) -> DSBlockMatrix3<T> {
        self.as_data()
            .view()
            .pruned(
                |a, b| *a.as_mut_arrays().as_mut_tensor() += b.into_arrays().as_tensor(),
                |i, j, e| keep(i, j, e.as_arrays().as_tensor()),
                |_, _| {},
            )
            .into_tensor()
    }
}

impl<T: Scalar> DSBlockMatrix1x3<T> {
    /// Assume that rows are monotonically increasing in the iterator. Columns don't have an order
    /// restriction.
    pub fn from_block_triplets_iter_uncompressed<I>(
        iter: I,
        num_rows: usize,
        num_cols: usize,
    ) -> Self
    where
        I: Iterator<Item = (usize, usize, [[T; 3]; 1])>,
    {
        let cap = iter.size_hint().0;
        let mut cols = Vec::with_capacity(cap);
        let mut blocks: Vec<[T; 3]> = Vec::with_capacity(cap);
        let mut offsets = Vec::with_capacity(num_rows);

        let mut prev_row = 0; // offset by +1 so we don't have to convert between isize.
        for (row, col, block) in iter {
            assert!(row + 1 >= prev_row); // We assume that rows are monotonically increasing.

            if row + 1 != prev_row {
                prev_row = row + 1;
                offsets.push(cols.len());
            }

            cols.push(col);
            // Push each row at a time to produce a Vec<[T; 3]>
            blocks.push(block[0]);
        }
        offsets.push(cols.len());
        offsets.shrink_to_fit();

        let mut col_data = Chunked::from_offsets(
            offsets,
            Sparse::from_dim(
                cols,
                num_cols,
                Chunked1::from_flat(Chunked3::from_array_vec(blocks)),
            ),
        );

        col_data.sort_chunks_by_index();

        col_data.into_tensor()
    }

    /// Assume that rows are monotonically increasing in the iterator. Columns don't have an order
    /// restriction.
    pub fn from_block_triplets_iter<I>(iter: I, num_rows: usize, num_cols: usize) -> Self
    where
        I: Iterator<Item = (usize, usize, [[T; 3]; 1])>,
    {
        Self::from_block_triplets_iter_uncompressed(iter, num_rows, num_cols).compressed()
    }
}

impl<T: Scalar, I: AsIndexSlice> DSBlockMatrix1x3<T, I> {
    /// Compress the matrix representation by consolidating duplicate entries.
    pub fn compressed(&self) -> DSBlockMatrix1x3<T> {
        self.as_data()
            .view()
            .compressed(|a, b| *a.as_mut_arrays().as_mut_tensor() += b.into_arrays().as_tensor())
            .into_tensor()
    }
}

impl<T: Scalar, I: AsIndexSlice> DSBlockMatrix1x3<T, I> {
    /// Remove all elements that do not satisfy the given predicate and compress the resulting matrix.
    pub fn pruned(
        &self,
        keep: impl Fn(usize, usize, &Matrix1x3<T>) -> bool,
    ) -> DSBlockMatrix1x3<T> {
        self.as_data()
            .view()
            .pruned(
                |a, b| *a.as_mut_arrays().as_mut_tensor() += b.into_arrays().as_tensor(),
                |i, j, e| keep(i, j, e.as_arrays().as_tensor()),
                |_, _| {},
            )
            .into_tensor()
    }
}

impl DSBlockMatrix3x1 {
    /// Assume that rows are monotonically increasing in the iterator. Columns don't have an order
    /// restriction.
    pub fn from_block_triplets_iter_uncompressed<I>(
        iter: I,
        num_rows: usize,
        num_cols: usize,
    ) -> Self
    where
        I: Iterator<Item = (usize, usize, [f64; 3])>,
    {
        let cap = iter.size_hint().0;
        let mut cols = Vec::with_capacity(cap);
        let mut values: Vec<f64> = Vec::with_capacity(cap*3);
        let mut offsets = Vec::with_capacity(num_rows);

        let mut prev_row = 0; // offset by +1 so we don't have to convert between isize.
        for (row, col, block) in iter {
            assert!(row + 1 >= prev_row); // We assume that rows are monotonically increasing.

            if row + 1 != prev_row {
                prev_row = row + 1;
                offsets.push(cols.len());
            }

            cols.push(col);
            // Push each row at a time to produce a Vec<f64>
            values.push(block[0]);
            values.push(block[1]);
            values.push(block[2]);
        }
        offsets.push(cols.len());
        offsets.shrink_to_fit();

        let mut col_data = Chunked::from_offsets(
            offsets,
            Sparse::from_dim(
                cols,
                num_cols,
                Chunked3::from_flat(Chunked1::from_flat(values)),
            ),
        );

        col_data.sort_chunks_by_index();

        col_data.into_tensor()
    }

    /// Assume that rows are monotonically increasing in the iterator. Columns don't have an order
    /// restriction.
    pub fn from_block_triplets_iter<I>(iter: I, num_rows: usize, num_cols: usize) -> Self
    where
        I: Iterator<Item = (usize, usize, [f64; 3])>,
    {
        Self::from_block_triplets_iter_uncompressed(iter, num_rows, num_cols).compressed()
    }
}

impl<T: Scalar, I: AsIndexSlice> DSBlockMatrix3x1<T, I> {
    /// Compress the matrix representation by consolidating duplicate entries.
    pub fn compressed(&self) -> DSBlockMatrix3x1<T> {
        self.as_data()
            .view()
            .compressed(|a, b| *a.as_mut_arrays().as_mut_tensor() += b.into_arrays().as_tensor())
            .into_tensor()
    }
}

impl<T: Scalar, I: AsIndexSlice> DSBlockMatrix3x1<T, I> {
    /// Remove all elements that do not satisfy the given predicate and compress the resulting matrix.
    pub fn pruned(
        &self,
        keep: impl Fn(usize, usize, &Matrix3x1<T>) -> bool,
    ) -> DSBlockMatrix3x1<T> {
        self.as_data()
            .view()
            .pruned(
                |a, b| *a.as_mut_arrays().as_mut_tensor() += b.into_arrays().as_tensor(),
                |i, j, e| keep(i, j, e.as_arrays().as_tensor()),
                |_, _| {},
            )
            .into_tensor()
    }
}

impl<'a, T: Scalar, I: Set + AsIndexSlice> DSBlockMatrix3<T, I> {
    pub fn write_img<P: AsRef<std::path::Path>>(&'a self, path: P) {
        use image::ImageBuffer;

        let nrows = BlockMatrix::num_total_rows(&self.view());
        let ncols = self.view().num_total_cols();

        if nrows == 0 || ncols == 0 {
            return;
        }

        let ciel = 10.0; //jac.max();
        let floor = -10.0; //jac.min();

        let img = ImageBuffer::from_fn(ncols as u32, nrows as u32, |c, r| {
            let val = self.coeff(r as usize, c as usize).to_f64().unwrap();
            let color = if val > 0.0 {
                [255, (255.0 * val / ciel) as u8, 0]
            } else if val < 0.0 {
                [0, (255.0 * (1.0 + val / floor)) as u8, 255]
            } else {
                [255, 0, 255]
            };
            image::Rgb(color)
        });

        img.save(path.as_ref())
            .expect("Failed to save matrix image.");
    }

    /// Get the value in the matrix at the given coordinates.
    pub fn coeff(&'a self, r: usize, c: usize) -> T {
        let row = self.as_data().view().isolate(r / 3);
        row.selection
            .indices
            .binary_search(&(c / 3))
            .map(|idx| row.source.isolate(idx).at(r % 3)[c % 3])
            .unwrap_or(T::zero())
    }
}

impl<T: Scalar> From<DBlockMatrix3<T>> for DSBlockMatrix3<T> {
    fn from(dense: DBlockMatrix3<T>) -> DSBlockMatrix3<T> {
        let num_rows = dense.num_rows();
        let num_cols = dense.num_cols();
        Chunked::from_sizes(
            vec![num_cols; num_rows], // num_cols blocks for every row
            Sparse::from_dim(
                (0..num_cols).cycle().take(num_cols * num_rows).collect(), // No sparsity
                num_cols,
                dense.into_data().data,
            ),
        )
        .into_tensor()
    }
}

impl<T: Scalar, I: AsRef<[usize]>> From<DiagonalBlockMatrix3<T, I>> for DSBlockMatrix3<T> {
    fn from(diag: DiagonalBlockMatrix3<T, I>) -> DSBlockMatrix3<T> {
        // Need to convert each triplet in diag into a diagonal 3x3 matrix.
        // Each block is essentially [x, 0, 0, 0, y, 0, 0, 0, z].
        let data: Chunked3<Vec<_>> = diag
            .view()
            .0
            .iter()
            .map(|&[x, y, z]| {
                [
                    [x, T::zero(), T::zero()],
                    [T::zero(), y, T::zero()],
                    [T::zero(), T::zero(), z],
                ]
            })
            .collect();

        let num_cols = diag.num_cols();
        Chunked::from_sizes(
            vec![1; diag.num_rows()], // One block in every row.
            Sparse::from_dim(
                (0..num_cols).collect(), // Diagonal sparsity pattern
                num_cols,
                Chunked3::from_flat(Chunked3::from_array_vec(data.into_storage())),
            ),
        )
        .into_tensor()
    }
}

impl<T: Scalar> From<DiagonalMatrix<T>> for DSMatrix<T> {
    fn from(diag: DiagonalMatrix<T>) -> DSMatrix<T> {
        let mut out_data = vec![T::zero(); diag.0.len()];
        Subset::clone_into_other(&diag.0.view(), &mut out_data);

        let num_cols = diag.num_cols();
        Chunked::from_sizes(
            vec![1; diag.num_rows()], // One block in every row.
            Sparse::from_dim(
                (0..num_cols).collect(), // Diagonal sparsity pattern
                num_cols,
                out_data,
            ),
        )
        .into_tensor()
    }
}

impl<T: Scalar> From<BlockDiagonalMatrix2<T>> for DSBlockMatrix2<T> {
    fn from(diag: BlockDiagonalMatrix2<T>) -> DSBlockMatrix2<T> {
        let mut out_data =
            Chunked2::from_flat(Chunked2::from_flat(vec![T::zero(); diag.0.len() * 4]));
        Subset::clone_into_other(&diag.0.view(), &mut out_data);

        let num_cols = diag.num_cols();
        Chunked::from_sizes(
            vec![1; diag.num_rows()], // One block in every row.
            Sparse::from_dim(
                (0..num_cols).collect(), // Diagonal sparsity pattern
                num_cols,
                out_data,
            ),
        )
        .into_tensor()
    }
}

/*
 * The following is an attempt at generic implementation of the function below
impl<'a, N> DSBlockMatrixView<'a, U<N>, U<N>>
where
    N: Copy + Default + Unsigned + Array<f64> + Array<<N as Array<f64>>::Array> + std::ops::Mul<N>,
    <N as Mul<N>>::Output: Copy + Default + Unsigned + Array<f64>,
{
    /// A special congruence tranform of the form `A -> P^T A P` where `P` is a block diagonal matrix.
    /// This is useful for instance as a change of basis transformation.
    fn diagonal_congruence_transform<M>(self, p: BlockDiagonalMatrixView<U<N>, U<M>>) -> DSBlockMatrix<U<M>, U<M>>
        where
            N: Array<<M as Array<f64>>::Array> + std::ops::Mul<M>,
            M: Copy + Default + Unsigned + Array<f64> + Array<<M as Array<f64>>::Array> + std::ops::Mul<M>,
            <M as Mul<M>>::Output: Copy + Default + Unsigned + Array<f64>,
            StaticRange<<N as std::ops::Mul<M>>::Output>: for<'b> IsolateIndex<&'b [f64]>,
    {
        let mut out = Chunked::from_offsets(
            (*self.data.offsets()).into_owned().into_inner(),
            Sparse::new(
                self.data.data().selection().clone().into_owned(),
                UniChunked::from_flat(UniChunked::from_flat(vec![0.0; self.num_non_zero_blocks() * 4]))
            ),
        );

        for (out_row, row) in Iterator::zip(out.iter_mut(), self.iter()) {
            for ((col_idx, out_entry), orig_entry) in out_row.indexed_source_iter_mut().zip(IntoIterator::into_iter(row.source().view())) {
                *out_entry.as_mut_tensor() = p.0.view().isolate(col_idx).as_tensor().transpose() * orig_entry.as_tensor() * p.0.view().isolate(col_idx).as_tensor();
            }
        }

        out.into_tensor()
    }
}
*/

impl<'a, T: Scalar> DSBlockMatrix3View<'a, T> {
    /// A special congruence tranform of the form `A -> P^T A P` where `P` is a block diagonal matrix.
    /// This is useful for instance as a change of basis transformation.
    pub fn diagonal_congruence_transform(
        self,
        p: BlockDiagonalMatrix3x2View<T>,
    ) -> DSBlockMatrix2<T> {
        let mut out = Chunked::from_offsets(
            (*self.as_data().offsets()).into_owned().into_inner(),
            Sparse::new(
                self.as_data().data().selection().clone().into_owned(),
                Chunked2::from_flat(Chunked2::from_flat(vec![
                    T::zero();
                    self.num_non_zero_blocks() * 4
                ])),
            ),
        );

        // TODO: It is annoying to always have to call into_arrays to construct small matrices.
        // We should implement array math on UniChunked<Array> types or find another solution.
        for (row_idx, (mut out_row, row)) in
            Iterator::zip(out.iter_mut(), self.as_data().iter()).enumerate()
        {
            for ((col_idx, out_entry), orig_entry) in out_row
                .indexed_source_iter_mut()
                .zip(IntoIterator::into_iter(row.source().view()))
            {
                let basis_lhs = *p.0.isolate(row_idx).into_arrays().as_tensor();
                let basis_rhs = *p.0.isolate(col_idx).into_arrays().as_tensor();
                let basis_lhs_tr = basis_lhs.transpose();
                let orig_block = *orig_entry.into_arrays().as_tensor();
                *out_entry.into_arrays().as_mut_tensor() = basis_lhs_tr * orig_block * basis_rhs;
            }
        }

        out.into_tensor()
    }
}

impl<'a, T: Scalar> DSBlockMatrix3View<'a, T> {
    /// A special congruence tranform of the form `A -> P^T A P` where `P` is a block diagonal matrix.
    /// This is useful for instance as a change of basis transformation.
    pub fn diagonal_congruence_transform3x1(self, p: BlockDiagonalMatrix3x1View<T>) -> DSMatrix<T> {
        let mut out = Chunked::from_offsets(
            (*self.as_data().offsets()).into_owned().into_inner(),
            Sparse::new(
                self.as_data().data().selection().clone().into_owned(),
                vec![T::zero(); self.num_non_zero_blocks()],
            ),
        );

        // TODO: It is annoying to always have to call into_arrays to construct small matrices.
        // We should implement array math on UniChunked<Array> types or find another solution.
        for (row_idx, (mut out_row, row)) in out.iter_mut().zip(self.as_data().iter()).enumerate() {
            for ((col_idx, out_entry), orig_entry) in out_row
                .indexed_source_iter_mut()
                .zip(IntoIterator::into_iter(row.source().view()))
            {
                let basis_lhs = *p.0.view().isolate(row_idx).into_arrays().as_tensor();
                let basis_rhs = *p.0.view().isolate(col_idx).into_arrays().as_tensor();
                let basis_lhs_tr = basis_lhs.transpose();
                let orig_block = *orig_entry.into_arrays().as_tensor();
                *out_entry = (basis_lhs_tr * orig_block * basis_rhs)[0][0];
            }
        }

        out.into_tensor()
    }
}

impl<'a, T: Scalar> Mul<&'a Tensor<[T]>> for DSMatrixView<'_, T> {
    type Output = Tensor<Vec<T>>;
    fn mul(self, rhs: &'a Tensor<[T]>) -> Self::Output {
        let mut res = vec![T::zero(); self.num_rows()].into_tensor();
        self.add_mul_in_place(rhs, res.view_mut());
        res
    }
}

impl<'a, T: Scalar> Mul<&'a Tensor<[T]>> for Transpose<DSMatrixView<'_, T>> {
    type Output = Tensor<Vec<T>>;
    fn mul(self, rhs: &'a Tensor<[T]>) -> Self::Output {
        let mut res = vec![T::zero(); self.num_rows()].into_tensor();
        assert_eq!(rhs.len(), self.num_cols());
        assert_eq!(res.len(), self.num_rows());
        let view = self.0.as_data();
        for (row_idx, (row, out_row)) in view.iter().zip(res.data.iter_mut()).enumerate() {
            let rhs_val = rhs.data[row_idx];
            for (_, mat_entry, _) in row.iter() {
                *out_row += *mat_entry * rhs_val;
            }
        }
        res
    }
}

const SMALLEST_CHUNK_SIZE: usize = 1_000;

impl<'a, T: Scalar> DSMatrixView<'_, T> {
    /// Parallel version of `add_mul_in_place`.
    pub fn add_left_mul_in_place_par(&self, lhs: &Tensor<[T]>, out: &mut Tensor<[T]>) {
        let nrows = self.num_rows();
        let ncols = self.num_cols();

        // Perform non-parallel multiply if there are too few rows.
        if nrows < SMALLEST_CHUNK_SIZE {
            return self.add_left_mul_in_place(lhs, out);
        }

        // Split rows into chunks.
        let mut nchunks = rayon::current_num_threads();
        let mut chunk_size = nrows / nchunks;
        if chunk_size < SMALLEST_CHUNK_SIZE {
            chunk_size = SMALLEST_CHUNK_SIZE.min(nrows);
            nchunks = nrows / chunk_size;
        }
        let last_chunk_size = nrows % nchunks;

        assert_eq!(lhs.len(), nrows);
        assert_eq!(out.len(), ncols);

        // Create nchunks (+1 for the last chunk) out_vectors to write to.
        let view = self.as_data().view();
        let clumped = Clumped::from_sizes_and_counts(
            vec![chunk_size, last_chunk_size],
            vec![nchunks, 1],
            view,
        );

        let out_vecs: Vec<Vec<T>> = clumped
            .par_iter()
            .zip(lhs.data.par_chunks_exact(chunk_size))
            .fold(
                || vec![T::zero(); ncols],
                |mut out, (m, lhs)| {
                    for (row, &lhs) in m.iter().zip(lhs.iter()) {
                        for (col_idx, &val, _) in row.iter() {
                            out[col_idx] += val * lhs;
                        }
                    }
                    out
                },
            )
            .collect();

        // Process the last chunk.
        for (row, &lhs) in clumped
            .iter()
            .nth(nchunks)
            .unwrap()
            .iter()
            .zip(lhs.data[chunk_size * nchunks..].iter())
        {
            for (col_idx, &val, _) in row.iter() {
                out.as_mut_data()[col_idx] += val * lhs;
            }
        }

        for v in out_vecs.iter() {
            *out += v.as_tensor();
        }
    }

    /// Parallel version of `add_mul_in_place`.
    pub fn add_mul_in_place_par(&self, rhs: &Tensor<[T]>, out: &mut Tensor<[T]>) {
        assert_eq!(rhs.len(), self.num_cols());
        assert_eq!(out.len(), self.num_rows());
        let view = self.as_data();

        view.par_iter()
            .zip(out.data.par_iter_mut())
            .for_each(move |(row, out)| {
                row.iter().for_each(move |(col_idx, entry, _)| {
                    *out += *entry * rhs.data[col_idx];
                })
            });
    }

    /// Adds the result of multiplying `self` by `lhs` on the left to `out`.
    pub fn add_left_mul_in_place(&self, lhs: &Tensor<[T]>, out: &mut Tensor<[T]>) {
        assert_eq!(lhs.len(), self.num_rows());
        assert_eq!(out.len(), self.num_cols());
        let view = self.as_data();

        for (row, &lhs) in view.iter().zip(lhs.data.iter()) {
            for (col_idx, &val, _) in row.iter() {
                out.data[col_idx] += val * lhs;
            }
        }
    }

    /// Adds the result of multiplying `self` by `rhs` on the right to `out`.
    pub fn add_mul_in_place(&self, rhs: &Tensor<[T]>, out: &mut Tensor<[T]>) {
        assert_eq!(rhs.len(), self.num_cols());
        assert_eq!(out.len(), self.num_rows());
        let view = self.as_data();

        for (row, out_row) in view.iter().zip(out.data.iter_mut()) {
            for (col_idx, entry, _) in row.iter() {
                *out_row += *entry * rhs.data[col_idx];
            }
        }
    }
}

impl<'a, T: Scalar> Mul<Tensor<Chunked3<&'a Tensor<[T]>>>> for DSBlockMatrix3View<'_, T> {
    type Output = Tensor<Chunked3<Tensor<Vec<T>>>>;
    fn mul(self, rhs: Tensor<Chunked3<&'a Tensor<[T]>>>) -> Self::Output {
        let rhs = rhs.as_data();
        assert_eq!(rhs.len(), self.num_cols());

        let mut res = Chunked3::from_array_vec(vec![[T::zero(); 3]; self.num_rows()]);
        for (row, out_row) in self.as_data().iter().zip(res.iter_mut()) {
            for (col_idx, block, _) in row.iter() {
                *out_row.as_mut_tensor() +=
                    *block.into_arrays().as_tensor() * rhs[col_idx].into_tensor();
            }
        }
        res.into_tensor()
    }
}

impl Mul<DSBlockMatrix1x3View<'_, f64>> for DSMatrixView<'_, f64> {
    type Output = DSBlockMatrix1x3<f64>;
    fn mul(self, rhs: DSBlockMatrix1x3View<'_, f64>) -> Self::Output {
        let mut blocks = Chunked1::from_flat(Chunked3::from_flat(Vec::new()));
        let mut offsets = vec![0];
        offsets.reserve(self.data.len());
        let mut indices = Vec::new();
        indices.reserve(5 * rhs.as_data().len());
        let mut workspace_blocks = Vec::new();

        for row in self.as_data().iter() {
            let mut out_expr = row.expr().cwise_mul(rhs.as_data().expr());

            // Write out block sums into a dense vector of blocks
            if let Some(next) = out_expr.next() {
                workspace_blocks.resize(
                    next.expr.target_size(),
                    Matrix1x3::new([[f64::zero(); 3]; 1]),
                );
                for elem in next.expr {
                    workspace_blocks[elem.index] += elem.expr;
                }
                for next in out_expr {
                    for elem in next.expr {
                        workspace_blocks[elem.index] += elem.expr;
                    }
                }
            }

            // Condense the dense blocks into a sparse vector
            for (i, block) in workspace_blocks
                .iter_mut()
                .enumerate()
                .filter(|(_, b)| !b.is_zero())
            {
                indices.push(i);
                blocks.eval_extend(std::mem::replace(
                    block,
                    Matrix1x3::new([[f64::zero(); 3]; 1]),
                ));
            }

            offsets.push(blocks.len());
        }
        let data = Sparse::from_dim(indices, rhs.num_cols(), blocks);
        Chunked::from_offsets(offsets, data).into_tensor()
    }
}

impl<'a, Rhs, T: Scalar> Mul<Rhs> for DSBlockMatrix1x3View<'_, T>
where
    Rhs: Into<Tensor<SubsetView<'a, Tensor<Chunked3<&'a Tensor<[T]>>>>>>,
{
    type Output = Tensor<Vec<T>>;
    fn mul(self, rhs: Rhs) -> Self::Output {
        let rhs = rhs.into();
        let rhs_data = rhs.into_data();
        assert_eq!(rhs_data.len(), self.num_cols());

        let mut res = vec![T::zero(); self.num_rows()];
        for (row, out_row) in self.as_data().iter().zip(res.iter_mut()) {
            for (col_idx, block, _) in row.iter() {
                *out_row.as_mut_tensor() +=
                    (*block.into_arrays().as_tensor() * Vector3::new(rhs_data[col_idx])).data[0];
            }
        }
        res.into_tensor()
    }
}

impl<T: Scalar> Mul<Tensor<Chunked3<&Tensor<[T]>>>> for DiagonalBlockMatrix3View<'_, T> {
    type Output = Tensor<Chunked3<Tensor<Vec<T>>>>;
    fn mul(self, other: Tensor<Chunked3<&Tensor<[T]>>>) -> Self::Output {
        let mut out = other.into_data().into_owned();
        for (&b, out) in self.0.iter().zip(out.iter_mut()) {
            for j in 0..3 {
                out[j] *= b[j];
            }
        }
        out.into_tensor()
    }
}

impl<T: Scalar> Mul<Tensor<Chunked3<&Tensor<[T]>>>> for DiagonalBlockMatrix3<T> {
    type Output = Tensor<Chunked3<Tensor<Vec<T>>>>;
    fn mul(self, other: Tensor<Chunked3<&Tensor<[T]>>>) -> Self::Output {
        let mut out = other.into_data().into_owned();
        for (&b, out) in self.0.iter().zip(out.iter_mut()) {
            for j in 0..3 {
                out[j] *= b[j];
            }
        }
        out.into_tensor()
    }
}

impl<'a, Rhs, T: Scalar> Mul<Rhs> for Transpose<BlockDiagonalMatrix3x1View<'a, T>>
where
    Rhs: Into<Tensor<SubsetView<'a, Tensor<Chunked3<&'a Tensor<[T]>>>>>>,
{
    type Output = Tensor<Vec<T>>;
    fn mul(self, rhs: Rhs) -> Self::Output {
        let rhs = rhs.into();
        let rhs_data = rhs.as_data();
        assert_eq!(rhs_data.len(), self.0.num_rows());

        let mut res = vec![T::zero(); self.0.num_cols()];
        for (idx, block) in (self.0).0.iter().enumerate() {
            res[idx] +=
                ((*block.into_arrays().as_tensor()).transpose() * Vector3::new(rhs_data[idx]))[0];
        }

        res.into_tensor()
    }
}

// A row vector of row-major 3x3 matrix blocks.
// This can also be interpreted as a column vector of column-major 3x3 matrix blocks.
pub type SparseBlockVectorBase<S, I = Vec<usize>, N = usize, M = usize> =
    Tensor<Sparse<Tensor<UniChunked<Tensor<UniChunked<S, M>>, N>>, Dim, I>>;
pub type SparseBlockVector<T = f64, I = Vec<usize>, N = usize, M = usize> =
    SparseBlockVectorBase<Tensor<Vec<T>>, I, N, M>;
pub type SparseBlockVectorView<'a, T = f64, N = usize, M = usize> =
    SparseBlockVectorBase<&'a Tensor<[T]>, &'a [usize], N, M>;
pub type SparseBlockVectorViewMut<'a, T = f64, N = usize, M = usize> =
    SparseBlockVectorBase<&'a mut Tensor<[T]>, &'a [usize], N, M>;
pub type SparseBlockVector3<T = f64, I = Vec<usize>> = SparseBlockVector<T, I, U3, U3>;
pub type SparseBlockVector3View<'a, T = f64> = SparseBlockVectorView<'a, T, U3, U3>;
pub type SparseBlockVector3ViewMut<'a, T = f64> = SparseBlockVectorViewMut<'a, T, U3, U3>;

/// A transpose of a matrix.
pub struct Transpose<M>(pub M);

impl<M: BlockMatrix> BlockMatrix for Transpose<M> {
    fn num_total_cols(&self) -> usize {
        self.0.num_total_rows()
    }
    fn num_total_rows(&self) -> usize {
        self.0.num_total_cols()
    }
    fn num_cols_per_block(&self) -> usize {
        self.0.num_rows_per_block()
    }
    fn num_rows_per_block(&self) -> usize {
        self.0.num_cols_per_block()
    }
}

impl<M: Matrix> Matrix for Transpose<M> {
    type Transpose = M;
    fn transpose(self) -> Self::Transpose {
        self.0
    }
    fn num_cols(&self) -> usize {
        self.0.num_rows()
    }
    fn num_rows(&self) -> usize {
        self.0.num_cols()
    }
}

impl<T, M> Norm<T> for Transpose<M>
where
    M: Norm<T>,
{
    fn lp_norm(&self, norm: LpNorm) -> T
    where
        T: Float,
    {
        self.0.lp_norm(norm)
    }
    fn norm_squared(&self) -> T {
        self.0.norm_squared()
    }
}

impl<'a, M: View<'a>> View<'a> for Transpose<M> {
    type Type = Transpose<M::Type>;
    fn view(&'a self) -> Self::Type {
        Transpose(self.0.view())
    }
}
impl<'a, M: ViewMut<'a>> ViewMut<'a> for Transpose<M> {
    type Type = Transpose<M::Type>;
    fn view_mut(&'a mut self) -> Self::Type {
        Transpose(self.0.view_mut())
    }
}

impl<M: Viewed> Viewed for Transpose<M> {}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

    /// Generate a random vector of float values between -1 and 1.
    pub fn random_vec<T>(n: usize, low: T, high: T) -> Vec<T>
    where
        T: Send + Sync + num_traits::Zero + Clone + rand::distributions::uniform::SampleUniform,
        T::Sampler: Sync,
        rand::distributions::Uniform<T>: Copy,
    {
        use rand::distributions::Uniform;
        use rand::prelude::*;
        let range = Uniform::new(low, high);
        let mut v = vec![T::zero(); n];
        v.par_chunks_mut(1.max(n / rayon::current_num_threads()))
            .for_each_init(
                || SeedableRng::from_seed([3; 32]),
                |rng: &mut StdRng, chunk| {
                    for c in chunk.iter_mut() {
                        *c = rng.sample(range);
                    }
                },
            );
        v
    }

    #[cfg(not(debug_assertions))]
    #[test]
    fn add_mul_in_place_par_stress() {
        let mut seq_t = std::time::Duration::new(0, 0);
        let mut par_t = std::time::Duration::new(0, 0);

        // Testing different sizes to make sure the chunking logic in
        // add_mul_in_place_par works.
        for n in 1_000_000..1_000_001 {
            for m in 1_000_000..1_000_001 {
                let mat_data = random_vec(100 * n, -1.0, 1.0);
                let mut mat_rows = random_vec(99 * n, 0, n);
                mat_rows.extend(0..n);
                let mat_cols = random_vec(100 * n, 0, m);
                let mat = DSMatrix::from_triplets_iter(
                    zip!(
                        mat_rows.into_iter(),
                        mat_cols.into_iter(),
                        mat_data.into_iter(),
                    ),
                    n,
                    m,
                );
                let rhs = random_vec(m, -1.0, 1.0);
                let mut out = vec![0.0; n];
                let mut out_par = vec![0.0; n];
                let par_now = std::time::Instant::now();
                mat.view()
                    .add_mul_in_place_par(rhs.as_tensor(), out_par.as_mut_tensor());
                par_t += par_now.elapsed();
                let seq_now = std::time::Instant::now();
                mat.view()
                    .add_mul_in_place(rhs.as_tensor(), out.as_mut_tensor());
                seq_t += seq_now.elapsed();
                assert!(out
                    .iter()
                    .zip(out_par.iter())
                    .all(|(&a, &b)| (a - b).abs() < 1e-8));
            }
        }
        std::dbg!(seq_t);
        std::dbg!(par_t);
    }

    #[cfg(not(debug_assertions))]
    #[test]
    fn add_left_mul_in_place_par_stress() {
        let mut seq_t = std::time::Duration::new(0, 0);
        let mut par_t = std::time::Duration::new(0, 0);
        rayon::ThreadPoolBuilder::new().build_global().unwrap();

        // Testing different sizes to make sure the chunking logic in
        // add_left_mul_in_place_par works.
        for n in 1_000_000..1_000_001 {
            for m in 1_000_000..1_000_001 {
                let mat_data = random_vec(100 * n, -1.0, 1.0);
                let mut mat_rows = random_vec(99 * n, 0, n);
                mat_rows.extend(0..n);
                let mat_cols = random_vec(100 * n, 0, m);
                let mat = DSMatrix::from_triplets_iter(
                    zip!(
                        mat_rows.into_iter(),
                        mat_cols.into_iter(),
                        mat_data.into_iter(),
                    ),
                    n,
                    m,
                );
                let lhs = random_vec(n, -1.0, 1.0);
                let mut out = vec![0.0; m];
                let mut out_par = vec![0.0; m];
                let par_now = std::time::Instant::now();
                mat.view()
                    .add_left_mul_in_place_par(lhs.as_tensor(), out_par.as_mut_tensor());
                par_t += par_now.elapsed();
                let seq_now = std::time::Instant::now();
                mat.view()
                    .add_left_mul_in_place(lhs.as_tensor(), out.as_mut_tensor());
                seq_t += seq_now.elapsed();
                assert!(out
                    .iter()
                    .zip(out_par.iter())
                    .all(|(&a, &b)| (a - b).abs() < 1e-8));
            }
        }
        std::dbg!(seq_t);
        std::dbg!(par_t);
    }
    #[test]
    fn add_left_mul_in_place_par() {
        // Build a 3x3 sparse matrix.
        // [1.0 2.0  . ]
        // [ .   .  3.0]
        // [ .   .  4.0]
        let mat_data = vec![1.0, 2.0, 3.0, 4.0];
        let mat_rows = vec![0, 0, 1, 2];
        let mat_cols = vec![0, 1, 2, 2];
        let mat = DSMatrix::from_triplets_iter(
            zip!(
                mat_rows.into_iter(),
                mat_cols.into_iter(),
                mat_data.into_iter(),
            ),
            3,
            3,
        );
        // Build left-hand-side vector.
        let lhs = vec![1.0, 2.0, 3.0];
        // Result should be [1.0, 2.0, 18.0]
        let expected = vec![1.0, 2.0, 18.0];
        let mut out = vec![0.0; 3];
        let mut out_par = vec![0.0; 3];
        mat.view()
            .add_left_mul_in_place(lhs.as_tensor(), out.as_mut_tensor());
        mat.view()
            .add_left_mul_in_place_par(lhs.as_tensor(), out_par.as_mut_tensor());
        assert!(
            out.iter()
                .zip(out_par.iter())
                .zip(expected.iter())
                .all(|((&a, &b), &expected)| a == expected && b == expected),
            "{:?} {:?}",
            &out,
            &out_par
        );

        // Testing different sizes to make sure the chunking logic in
        // add_left_mul_in_place_par works.
        for n in 10..50 {
            for m in 10..50 {
                let mat_data = random_vec(5 * n, -1.0, 1.0);
                let mut mat_rows = random_vec(4 * n, 0, n);
                mat_rows.extend(0..n);
                let mat_cols = random_vec(5 * n, 0, m);
                let mat = DSMatrix::from_triplets_iter(
                    zip!(
                        mat_rows.into_iter(),
                        mat_cols.into_iter(),
                        mat_data.into_iter(),
                    ),
                    n,
                    m,
                );
                let lhs = random_vec(n, -1.0, 1.0);
                let mut out = vec![0.0; m];
                let mut out_par = vec![0.0; m];
                mat.view()
                    .add_left_mul_in_place(lhs.as_tensor(), out.as_mut_tensor());
                mat.view()
                    .add_left_mul_in_place_par(lhs.as_tensor(), out_par.as_mut_tensor());
                assert!(
                    out.iter()
                        .zip(out_par.iter())
                        .all(|(&a, &b)| (a - b).abs() < 1e-2),
                    "{:?} {:?}",
                    &out,
                    &out_par
                );
            }
        }
    }

    #[test]
    fn add_mul_in_place_par() {
        // Testing different sizes to make sure the chunking logic in
        // add_mul_in_place_par works.
        for n in 10..50 {
            for m in 10..50 {
                let mat_data = random_vec(5 * n, -1.0, 1.0);
                let mut mat_rows = random_vec(4 * n, 0, n);
                mat_rows.extend(0..n);
                let mat_cols = random_vec(5 * n, 0, m);
                let mat = DSMatrix::from_triplets_iter(
                    zip!(
                        mat_rows.into_iter(),
                        mat_cols.into_iter(),
                        mat_data.into_iter(),
                    ),
                    n,
                    m,
                );
                let rhs = random_vec(m, -1.0, 1.0);
                let mut out = vec![0.0; n];
                let mut out_par = vec![0.0; n];
                mat.view()
                    .add_mul_in_place(rhs.as_tensor(), out.as_mut_tensor());
                mat.view()
                    .add_mul_in_place_par(rhs.as_tensor(), out_par.as_mut_tensor());
                assert!(out
                    .iter()
                    .zip(out_par.iter())
                    .all(|(&a, &b)| (a - b).abs() < 1e-8));
            }
        }
    }

    #[test]
    fn sparse_sparse_mul_diag() {
        let blocks = vec![
            // Block 1
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            // Block 2
            [1.1, 2.2, 3.3],
            [4.4, 5.5, 6.6],
            [7.7, 8.8, 9.9],
        ];
        let chunked_blocks = Chunked3::from_flat(Chunked3::from_array_vec(blocks));
        let indices = vec![(1, 1), (3, 2)];
        let mtx =
            SSBlockMatrix3::from_index_iter_and_data(indices.iter().cloned(), 4, 3, chunked_blocks);

        let sym = mtx.view() * mtx.view().transpose();

        let exp_vec = vec![
            14.0, 32.0, 50.0, 32.0, 77.0, 122.0, 50.0, 122.0, 194.0, 16.94, 38.72, 60.5, 38.72,
            93.17, 147.62, 60.5, 147.62, 234.74,
        ];

        let val_vec = sym.storage();
        for (&val, &exp) in val_vec.iter().zip(exp_vec.iter()) {
            assert_relative_eq!(val, exp);
        }
    }

    #[test]
    fn sparse_diag_add() {
        let blocks = vec![
            // Block 1
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            // Block 2
            [1.1, 2.2, 3.3],
            [4.4, 5.5, 6.6],
            [7.7, 8.8, 9.9],
        ];
        let chunked_blocks = Chunked3::from_flat(Chunked3::from_array_vec(blocks));
        let indices = vec![(1, 1), (3, 2)];
        let mtx =
            SSBlockMatrix3::from_index_iter_and_data(indices.iter().cloned(), 4, 3, chunked_blocks);

        let sym = mtx.view() * mtx.view().transpose();

        let diagonal: Vec<_> = (1..=12).map(|i| i as f64).collect();
        let diag = DiagonalBlockMatrix::from_uniform(Chunked3::from_flat(diagonal));

        let non_singular_sym = sym.view() + diag.view();

        let exp_vec = vec![
            1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 18.0, 32.0, 50.0, 32.0, 82.0, 122.0, 50.0,
            122.0, 200.0, 7.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 9.0, 26.94, 38.72, 60.5, 38.72,
            104.17, 147.62, 60.5, 147.62, 246.74,
        ];

        let val_vec = non_singular_sym.storage();
        for (&val, &exp) in val_vec.iter().zip(exp_vec.iter()) {
            assert_relative_eq!(val, exp);
        }
    }

    #[test]
    fn sparse_sparse_mul_non_diag() {
        let blocks = vec![
            // Block 1
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            // Block 2
            [10.0, 11.0, 12.0],
            [16.0, 17.0, 18.0],
            [22.0, 23.0, 24.0],
            // Block 3
            [13.0, 14.0, 15.0],
            [19.0, 20.0, 21.0],
            [25.0, 26.0, 27.0],
        ];
        let chunked_blocks = Chunked3::from_flat(Chunked3::from_array_vec(blocks));
        let indices = vec![(1, 0), (2, 0), (2, 1)];
        let mtx =
            SSBlockMatrix3::from_index_iter_and_data(indices.iter().cloned(), 3, 2, chunked_blocks);

        let sym = mtx.view() * mtx.view().transpose();

        let exp_vec = vec![
            14.0, 32.0, 50.0, 32.0, 77.0, 122.0, 50.0, 122.0, 194.0, 68.0, 104.0, 140.0, 167.0,
            257.0, 347.0, 266.0, 410.0, 554.0, 68.0, 167.0, 266.0, 104.0, 257.0, 410.0, 140.0,
            347.0, 554.0, 955.0, 1405.0, 1855.0, 1405.0, 2071.0, 2737.0, 1855.0, 2737.0, 3619.0,
        ];

        let val_vec = sym.storage();
        for (&val, &exp) in val_vec.iter().zip(exp_vec.iter()) {
            assert_relative_eq!(val, exp);
        }
    }

    #[test]
    fn sparse_sparse_mul_non_diag_uncompressed() {
        let blocks = vec![
            // Block 1
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            // Block 2
            [10.0, 11.0, 12.0],
            [16.0, 17.0, 18.0],
            [22.0, 23.0, 24.0],
            // Block 3
            [0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1],
            // Block 4
            [13.0, 14.0, 15.0],
            [19.0, 20.0, 21.0],
            [25.0, 26.0, 27.0],
        ];
        let chunked_blocks = Chunked3::from_flat(Chunked3::from_array_vec(blocks));
        let indices = vec![(1, 0), (2, 0), (2, 0), (2, 1)];
        let mtx =
            SSBlockMatrix3::from_index_iter_and_data(indices.iter().cloned(), 3, 2, chunked_blocks);

        let sym = mtx.view() * mtx.view().transpose();

        let exp_vec = vec![
            14.0, 32.0, 50.0, 32.0, 77.0, 122.0, 50.0, 122.0, 194.0, 68.6, 104.6, 140.6, 168.5,
            258.5, 348.5, 268.4, 412.4, 556.4, 68.6, 168.5, 268.4, 104.6, 258.5, 412.4, 140.6,
            348.5, 556.4, 961.63, 1413.43, 1865.23, 1413.43, 2081.23, 2749.03, 1865.23, 2749.03,
            3632.83,
        ];

        let val_vec = sym.storage();
        for (&val, &exp) in val_vec.iter().zip(exp_vec.iter()) {
            assert_relative_eq!(val, exp);
        }
    }

    //#[test]
    //fn ds_mtx_mul_ds_block_mtx_1x3() {
    //    // [1 2 .]
    //    // [. . 3]
    //    // [. 2 .]
    //    let ds = Chunked::from_sizes(vec![2, 1, 1], Sparse::from_dim(vec![0, 1, 2, 1], 3, vec![1.0,2.0,3.0,2.0]));
    //    // [1 2 3 . . .]
    //    // [1 2 1 4 5 6]
    //    // [7 8 9 . . .]
    //    let blocks = Chunked1::from_flat(Chunked3::from_flat(
    //            vec![1.0, 2.0, 3.0, 1.0, 2.0, 1.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]));
    //    let ds_block_mtx_1x3 = Chunked::from_sizes(vec![1, 2, 1], Sparse::from_dim(vec![0, 0, 1, 0], 2, blocks));

    //    let out = JaggedTensor::new(ds.view()) * JaggedTensor::new(ds_block_mtx_1x3.view());

    //    // [ 3  6  5  8 10 12]
    //    // [21 24 27  .  .  .]
    //    // [ 2  4  2  8 10 12]
    //    let exp_blocks = Chunked1::from_flat(Chunked3::from_flat(
    //            vec![3.0,6.0,5.0,8.0,10.0,12.0,21.0,24.0,27.0,2.0,4.0,2.0,8.0,10.0,12.0]));
    //    assert_eq!(out.data, Chunked::from_sizes(vec![2,1,2], Sparse::from_dim(vec![0,1,0,0,1], 2, exp_blocks)));
    //}
}
