//!
//! Sparse-row sparse-column block matrix.
//!

use super::*;

type Dim = std::ops::RangeTo<usize>;

/// Sparse-row sparse-column 3x3 block matrix.
pub type SSBlockMatrixBase<S = Vec<f64>, I = Vec<usize>, N = usize, M = usize> = Tensor<
    Sparse<
        Tensor<
            Chunked<
                Tensor<Sparse<Tensor<UniChunked<Tensor<UniChunked<S, M>>, N>>, Dim, I>>,
                Offsets<I>,
            >,
        >,
        Dim,
        I,
    >,
>;

pub type SSBlockMatrix<T = f64, I = Vec<usize>, N = usize, M = usize> =
    SSBlockMatrixBase<Tensor<Vec<T>>, I, N, M>;
pub type SSBlockMatrixView<'a, T = f64, N = usize, M = usize> =
    SSBlockMatrixBase<&'a Tensor<[T]>, &'a [usize], N, M>;
pub type SSBlockMatrix3<T = f64, I = Vec<usize>> = SSBlockMatrix<T, I, U3, U3>;
pub type SSBlockMatrix3View<'a, T = f64> = SSBlockMatrixView<'a, T, U3, U3>;
pub type SSBlockMatrix3x1<T = f64, I = Vec<usize>> = SSBlockMatrix<T, I, U3, U1>;
pub type SSBlockMatrix3x1View<'a, T = f64> = SSBlockMatrixView<'a, T, U3, U1>;

impl<S: Set + IntoData, I, N: Dimension, M: Dimension> BlockMatrix
    for SSBlockMatrixBase<S, I, N, M>
{
    fn num_cols_per_block(&self) -> usize {
        self.as_data().source().data().source().data().chunk_size()
    }
    fn num_rows_per_block(&self) -> usize {
        self.as_data().source().data().source().chunk_size()
    }
    fn num_total_cols(&self) -> usize {
        self.num_cols() * self.num_cols_per_block()
    }
    fn num_total_rows(&self) -> usize {
        self.num_rows() * self.num_rows_per_block()
    }
}

impl<S: Set + IntoData, I, N: Dimension, M: Dimension> Matrix for SSBlockMatrixBase<S, I, N, M> {
    type Transpose = Transpose<Self>;
    fn transpose(self) -> Self::Transpose {
        Transpose(self)
    }
    fn num_cols(&self) -> usize {
        self.as_data().source().data().selection().target.distance()
    }
    fn num_rows(&self) -> usize {
        self.as_data().selection().target.distance()
    }
}

// 3x3 block matrix impls

impl<'a, T: Scalar, I: Set + AsIndexSlice> SSBlockMatrix3<T, I> {
    pub fn write_img<P: AsRef<std::path::Path>>(&'a self, path: P) {
        use image::ImageBuffer;

        let nrows = self.num_total_rows();
        let ncols = self.num_total_cols();
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
        let view = self.view().into_data();
        if let Ok(row) = view
            .selection
            .indices
            .binary_search(&(r / 3))
            .map(|idx| view.source.isolate(idx))
        {
            row.selection
                .indices
                .binary_search(&(c / 3))
                .map(|idx| row.source.isolate(idx).at(r % 3)[c % 3])
                .unwrap_or_else(|_| T::zero())
        } else {
            T::zero()
        }
    }
}

impl<T: Scalar> SSBlockMatrix3<T> {
    pub fn from_index_iter_and_data<It: Iterator<Item = (usize, usize)>>(
        index_iter: It,
        num_rows: usize,
        num_cols: usize,
        blocks: Chunked3<Chunked3<Vec<T>>>,
    ) -> Self {
        Self::from_block_triplets_iter(
            index_iter
                .zip(blocks.iter())
                .map(|((i, j), x)| (i, j, *x.into_arrays())),
            num_rows,
            num_cols,
        )
    }

    /// Assume that rows are monotonically increasing in the iterator.
    pub fn from_block_triplets_iter_uncompressed<I>(
        iter: I,
        num_rows: usize,
        num_cols: usize,
    ) -> Self
    where
        I: Iterator<Item = (usize, usize, [[T; 3]; 3])>,
    {
        let cap = iter.size_hint().0;
        let mut rows = Vec::with_capacity(cap);
        let mut cols = Vec::with_capacity(cap);
        let mut blocks: Vec<[T; 3]> = Vec::with_capacity(cap);
        let mut offsets = Vec::with_capacity(num_rows);

        let mut prev_row = 0; // offset by +1 so we don't have to convert between isize.
        for (row, col, block) in iter {
            assert!(row + 1 >= prev_row); // We assume that rows are monotonically increasing.

            if row + 1 != prev_row {
                rows.push(row);
                prev_row = row + 1;
                offsets.push(cols.len());
            }

            cols.push(col);
            // Push each row at a time to produce a Vec<[T; 3]>
            blocks.push(block[0]);
            blocks.push(block[1]);
            blocks.push(block[2]);
        }
        offsets.push(cols.len());
        offsets.shrink_to_fit();
        rows.shrink_to_fit();

        let mut col_data = Chunked::from_offsets(
            offsets,
            Sparse::from_dim(
                cols,
                num_cols,
                Chunked3::from_flat(Chunked3::from_array_vec(blocks)),
            ),
        );

        col_data.sort_chunks_by_index();

        Sparse::from_dim(rows, num_rows, col_data).into_tensor()
    }
    pub fn from_block_triplets_iter<I>(iter: I, num_rows: usize, num_cols: usize) -> Self
    where
        I: Iterator<Item = (usize, usize, [[T; 3]; 3])>,
    {
        Self::from_block_triplets_iter_uncompressed(iter, num_rows, num_cols).compressed()
    }
}

fn is_unique(indices: &[usize]) -> bool {
    if indices.is_empty() {
        return true;
    }
    let mut prev_index = indices[0];
    for &index in indices[1..].iter() {
        if index == prev_index {
            return false;
        }
        prev_index = index;
    }
    true
}

impl<T: Scalar, I: AsIndexSlice> SSBlockMatrix3<T, I>
where
    Self: for<'a> View<'a, Type = SSBlockMatrix3View<'a, T>>,
    I: IntoOwned<Owned = Vec<usize>>,
{
    /// Compress the matrix representation by consolidating duplicate entries.
    pub fn compressed(&self) -> SSBlockMatrix3<T> {
        let data = self.as_data();
        // Check that there are no duplicate rows. This should not happen when crating from
        // triplets.
        assert!(is_unique(data.selection.indices.as_ref()));
        Sparse::new(
            data.selection.view().into_owned(),
            data.view().source.compressed(|a, b| {
                *AsMutTensor::as_mut_tensor(a.as_mut_arrays()) += b.into_arrays().as_tensor()
            }),
        )
        .into_tensor()
    }
}

impl<T: Scalar, I: AsIndexSlice> SSBlockMatrix3<T, I>
where
    Self: for<'a> View<'a, Type = SSBlockMatrix3View<'a, T>>,
    I: IntoOwned<Owned = Vec<usize>>,
{
    /// Remove all elements that do not satisfy the given predicate and compress the resulting matrix.
    pub fn pruned(&self, keep: impl Fn(usize, usize, &Matrix3<T>) -> bool) -> SSBlockMatrix3<T> {
        self.pruned_with(keep, |_, _| {})
    }

    /// Remove all elements that do not satisfy the given predicate and compress the resulting matrix.
    pub fn pruned_with(
        &self,
        keep: impl Fn(usize, usize, &Matrix3<T>) -> bool,
        mapping: impl FnMut(usize, usize),
    ) -> SSBlockMatrix3<T> {
        let data = self.as_data();
        // Check that there are no duplicate rows. This should not happen when crating from
        // triplets.
        assert!(is_unique(data.selection.indices.as_ref()));
        Sparse::new(
            data.selection.view().into_owned(),
            data.view().source.pruned(
                |a, b| *a.as_mut_arrays().as_mut_tensor() += b.into_arrays().as_tensor(),
                |i, j, e| keep(i, j, e.as_arrays().as_tensor()),
                mapping,
            ),
        )
        .into_tensor()
    }
}

impl<T: Scalar> Add<DiagonalBlockMatrix3View<'_, T>> for SSBlockMatrix3View<'_, T> {
    type Output = DSBlockMatrix3<T>;
    fn add(self, rhs: DiagonalBlockMatrix3View<'_, T>) -> Self::Output {
        let rhs = rhs.0;
        let num_rows = self.num_rows();
        let num_cols = self.num_cols();

        let lhs_nnz = self.as_data().source.data.source.len();
        let rhs_nnz = rhs.len();
        let num_non_zero_blocks = lhs_nnz + rhs_nnz;

        let mut non_zero_row_offsets = vec![num_non_zero_blocks; num_rows + 1];
        non_zero_row_offsets[0] = 0;

        let mut out = Chunked::from_offsets(
            non_zero_row_offsets,
            Sparse::from_dim(
                vec![0; num_non_zero_blocks], // Pre-allocate column index vec.
                num_cols,
                Chunked3::from_flat(Chunked3::from_flat(vec![
                    T::zero();
                    num_non_zero_blocks * 9
                ])),
            ),
        );

        let mut rhs_iter = rhs.iter().enumerate();

        let add_diagonal_entry = |out: Chunked3<&mut [T; 9]>, entry: &[T; 3]| {
            let out_mtx = out.into_arrays();
            out_mtx[0][0] += entry[0];
            out_mtx[1][1] += entry[1];
            out_mtx[2][2] += entry[2];
        };

        for (sparse_row_idx, row_l, _) in self.as_data().iter() {
            let (row_idx, rhs_entry) = loop {
                if let Some((row_idx, entry)) = rhs_iter.next() {
                    if row_idx < sparse_row_idx {
                        let out_row = out.view_mut().isolate(row_idx);
                        let (idx, out_col, _) = out_row.isolate(0);
                        *idx = row_idx; // Diagonal entry col_idx == row_idx
                        add_diagonal_entry(out_col, entry);
                        out.transfer_forward_all_but(row_idx, 1);
                    } else {
                        break (row_idx, entry);
                    }
                } else {
                    assert!(false, "RHS ran out of entries");
                }
            };

            assert_eq!(sparse_row_idx, row_idx);
            // Copy row from lhs and add or add to the diagonal entry.

            let mut count_out_cols = 0;
            let mut prev_col_idx = 0;
            for (col_idx, col, _) in row_l.iter() {
                if col_idx > row_idx && prev_col_idx < row_idx {
                    let out_row = out.view_mut().isolate(row_idx);
                    // Additional diagonal entry, add rhs entry before adding
                    // subsequent entries from lhs to preserve order of indices.
                    let (out_col_idx, out_col, _) = out_row.isolate(count_out_cols);
                    add_diagonal_entry(out_col, rhs_entry);
                    *out_col_idx = row_idx;
                    count_out_cols += 1;
                }

                let out_row = out.view_mut().isolate(row_idx);
                let (out_col_idx, mut out_col, _) = out_row.isolate(count_out_cols);
                out_col.copy_from_flat(*col.data());
                *out_col_idx = col_idx;
                if col_idx == row_idx {
                    add_diagonal_entry(out_col, rhs_entry);
                }

                prev_col_idx = col_idx;
                count_out_cols += 1;
            }

            // Truncate the current row to fit.
            out.transfer_forward_all_but(row_idx, count_out_cols);
        }
        out.trim();

        out.into_tensor()
    }
}

impl<'a, T: Scalar, Rhs> Mul<Rhs> for SSBlockMatrix3View<'_, T>
where
    Rhs: Into<Tensor<SubsetView<'a, Tensor<Chunked3<&'a Tensor<[T]>>>>>>,
{
    type Output = Tensor<Chunked3<Tensor<Vec<T>>>>;
    fn mul(self, rhs: Rhs) -> Self::Output {
        let rhs = rhs.into();
        let rhs_data = rhs.as_data();
        assert_eq!(rhs_data.len(), self.num_cols());

        let mut res = Chunked3::from_array_vec(vec![[T::zero(); 3]; self.num_rows()]);
        for (row_idx, row, _) in self.as_data().iter() {
            for (col_idx, block, _) in row.iter() {
                *res[row_idx].as_mut_tensor() +=
                    *block.into_arrays().as_tensor() * rhs_data[col_idx].into_tensor();
            }
        }

        res.into_tensor()
    }
}

impl<'a, T: Scalar> Mul<Tensor<SparseView<'a, Tensor<Chunked3<&'a Tensor<[T]>>>>>>
    for SSBlockMatrix3View<'a, T>
{
    type Output = Tensor<SparseView<'a, Tensor<Chunked3<Tensor<Vec<T>>>>>>;
    fn mul(self, rhs: Tensor<SparseView<'a, Tensor<Chunked3<&'a Tensor<[T]>>>>>) -> Self::Output {
        let rhs_data = rhs.as_data();
        assert_eq!(rhs_data.len(), self.num_cols());

        let mut res = Chunked3::from_flat(vec![T::zero(); self.data.len() * 3]);

        for (index, (_, row, _)) in self.as_data().iter().enumerate() {
            let mut lhs_val_iter = row.into_iter().peekable();
            let mut rhs_val_iter = rhs_data.into_iter().peekable();
            while lhs_val_iter.peek().is_some() || rhs_val_iter.peek().is_some() {
                if let (Some(left), Some(right)) = (lhs_val_iter.peek(), rhs_val_iter.peek()) {
                    if left.0 < right.0 {
                        let _ = lhs_val_iter.next().unwrap();
                    } else if left.0 > right.0 {
                        let _ = rhs_val_iter.next().unwrap();
                    } else {
                        let left = lhs_val_iter.next().unwrap().1.into_arrays();
                        let right = rhs_val_iter.next().unwrap().1.into_data().into_tensor();
                        *res[index].as_mut_tensor() += *left.as_tensor() * right;
                    }
                }
            }
        }

        Sparse::new(self.data.selection, res).into_tensor()
    }
}

impl<'a, T: Scalar, Rhs> Mul<Rhs> for Transpose<SSBlockMatrix3View<'_, T>>
where
    Rhs: Into<Tensor<SubsetView<'a, Tensor<Chunked3<&'a Tensor<[T]>>>>>>,
{
    type Output = Tensor<Chunked3<Tensor<Vec<T>>>>;
    fn mul(self, rhs: Rhs) -> Self::Output {
        let rhs = rhs.into();
        let rhs_data = rhs.as_data();
        assert_eq!(rhs_data.len(), self.num_cols());

        let mut res = Chunked3::from_array_vec(vec![[T::zero(); 3]; self.num_rows()]);
        for (col_idx, col, _) in self.0.as_data().iter() {
            let rhs = rhs_data[col_idx].into_tensor();
            for (row_idx, block, _) in col.iter() {
                *res[row_idx].as_mut_tensor() +=
                    (rhs.transpose() * *block.into_arrays().as_tensor())[0];
            }
        }

        res.into_tensor()
    }
}

impl<'a, T: Scalar> Mul<Tensor<SparseView<'a, Tensor<Chunked3<&'a Tensor<[T]>>>>>>
    for Transpose<SSBlockMatrix3View<'_, T>>
{
    type Output = Tensor<Chunked3<Tensor<Vec<T>>>>;
    fn mul(self, rhs: Tensor<SparseView<'a, Tensor<Chunked3<&'a Tensor<[T]>>>>>) -> Self::Output {
        let rhs_data = rhs.as_data();
        assert_eq!(rhs_data.len(), self.num_cols());

        let mut res = Chunked3::from_flat(vec![T::zero(); self.num_rows()]);
        let mut lhs_col_iter = self.0.as_data().iter().peekable();
        let mut rhs_val_iter = rhs_data.into_iter().peekable();

        while lhs_col_iter.peek().is_some() || rhs_val_iter.peek().is_some() {
            if let (Some(left), Some(right)) = (lhs_col_iter.peek(), rhs_val_iter.peek()) {
                if left.0 < right.0 {
                    let _ = lhs_col_iter.next().unwrap();
                } else if left.0 > right.0 {
                    let _ = rhs_val_iter.next().unwrap();
                } else {
                    let left = lhs_col_iter.next().unwrap().1;
                    let right = rhs_val_iter.next().unwrap().1;
                    for (index, left_block, _) in left.iter() {
                        let left_block = left_block.into_arrays();
                        for i in 0..3 {
                            *res[index].as_mut_tensor() += *left_block[i].as_tensor() * right[i];
                        }
                    }
                }
            }
        }

        res.into_tensor()
    }
}

impl<T: Scalar> MulAssign<DiagonalBlockMatrix3<T>> for SSBlockMatrix3<T> {
    fn mul_assign(&mut self, rhs: DiagonalBlockMatrix3<T>) {
        let rhs = View::view(&rhs);
        self.mul_assign(rhs);
    }
}

impl<T: Scalar> MulAssign<DiagonalBlockMatrix3View<'_, T>> for SSBlockMatrix3<T> {
    fn mul_assign(&mut self, rhs: DiagonalBlockMatrix3View<'_, T>) {
        assert_eq!(rhs.0.len(), self.num_cols());
        for (_, mut row) in self.view_mut().as_mut_data().iter_mut() {
            for (col_idx, mut block) in row.iter_mut() {
                let mass_vec = *rhs.0.at(*col_idx);
                for (block_row, &mass) in block.iter_mut().zip(mass_vec.iter()) {
                    *block_row = (Vector3::new(*block_row) * mass).into();
                }
            }
        }
    }
}

impl<T: Scalar> Mul<Transpose<SSBlockMatrix3View<'_, T>>> for SSBlockMatrix3View<'_, T> {
    type Output = SSBlockMatrix3<T>;
    fn mul(self, rhs: Transpose<SSBlockMatrix3View<T>>) -> Self::Output {
        let rhs_t = rhs.0;
        let num_rows = self.num_rows();
        let num_cols = rhs_t.num_rows();

        let lhs_nnz = self.as_data().source.data.source.len();
        let rhs_nnz = rhs_t.as_data().source.data.source.len();
        let num_non_zero_cols = rhs_t.as_data().indices().len();
        let num_non_zero_blocks = (lhs_nnz + rhs_nnz).max(num_non_zero_cols);

        // Allocate enough offsets for all non-zero rows in self. and assign the
        // first row to contain all elements by setting all offsets to
        // num_non_zero_blocks except the first.
        let mut non_zero_row_offsets = vec![num_non_zero_blocks; self.as_data().len() + 1];
        non_zero_row_offsets[0] = 0;

        let mut out = Sparse::from_dim(
            self.as_data().indices().to_vec(),
            num_rows,
            Chunked::from_offsets(
                non_zero_row_offsets,
                Sparse::from_dim(
                    vec![0; num_non_zero_blocks], // Pre-allocate column index vec.
                    num_cols,
                    Chunked3::from_flat(Chunked3::from_flat(vec![
                        T::zero();
                        num_non_zero_blocks * 9
                    ])),
                ),
            ),
        );

        let mut nz_row_idx = 0;
        for (row_idx, row_l, _) in self.as_data().iter() {
            let (_, out_row, _) = out.view_mut().isolate(nz_row_idx);
            let num_non_zero_blocks_in_row =
                rhs_t.mul_sparse_matrix3_vector(row_l.into_tensor(), out_row.into_tensor());

            // Truncate resulting row. This makes space for the next row in the output.
            if num_non_zero_blocks_in_row > 0 {
                // This row is non-zero, set the row index in the output.
                out.indices_mut()[nz_row_idx] = row_idx;
                // Truncate the current row to fit.
                out.source_mut()
                    .transfer_forward_all_but(nz_row_idx, num_non_zero_blocks_in_row);
                nz_row_idx += 1;
            }

            // We may run out of memory in out. Check this and allocate space for each additional
            // row as needed.
            if nz_row_idx < out.len() {
                let num_available = out.view().isolate(nz_row_idx).1.len();
                if num_available < num_non_zero_cols {
                    // The next row has less than num_non_zero_cols available space. We should allocate
                    // additional entries.
                    // First append entries to the last chunk.
                    let num_new_elements = num_non_zero_cols - num_available;
                    Chunked::extend_last(
                        &mut out.source_mut(),
                        std::iter::repeat((0, Chunked3::from_flat([T::zero(); 9])))
                            .take(num_new_elements),
                    );
                    // Next we transfer all elements of the last chunk into the current row.
                    for idx in (nz_row_idx + 1..out.len()).rev() {
                        out.source_mut().transfer_backward(idx, num_new_elements);
                    }
                }
            }
        }

        // There may be fewer non-zero rows than in self. Truncate those
        // and truncate the entries in storage we didn't use.
        out.trim();

        out.into_tensor()
    }
}

impl<T: Scalar> Transpose<SSBlockMatrix3<T>> {
    pub fn premultiply_block_diagonal_mtx(&mut self, lhs: BlockDiagonalMatrix3View<'_, T>) {
        assert_eq!(lhs.num_cols(), self.num_rows());
        for (_, mut col) in self.0.view_mut().as_mut_data().iter_mut() {
            for (row_idx, block) in col.iter_mut() {
                let lhs_block: Matrix3<T> = *lhs.0.at(*row_idx).into_arrays().as_tensor();
                let out_block = block.into_arrays().as_mut_tensor();
                let rhs_block = *out_block;
                out_block[0] = lhs_block * rhs_block[0];
                out_block[1] = lhs_block * rhs_block[1];
                out_block[2] = lhs_block * rhs_block[2];
            }
        }
    }
}

impl<T: Scalar> SSBlockMatrix3View<'_, T> {
    /// Multiply `self` by the given `rhs` vector into the given `out` view.
    ///
    /// Note that the output vector `out` may be more sparse than the number of
    /// rows in `self`, however it is assumed that enough elements is allocated
    /// in `out` to ensure that the result fits. Entries are packed towards the
    /// beginning of out, and the number of non-zeros produced is returned so it
    /// can be simply truncated to fit at the end of this function.
    fn mul_sparse_matrix3_vector(
        self,
        rhs: SparseBlockVector3View<T>,
        mut out: SparseBlockVectorBase<&mut Tensor<[T]>, &mut [usize], U3, U3>,
    ) -> usize {
        // The output iterator will advance when we see a non-zero result.
        let mut out_iter_mut = out.as_mut_data().iter_mut();
        let mut num_non_zeros = 0;

        for (row_idx, row, _) in self.as_data().iter() {
            // Initialize output
            let mut sum_mtx = Matrix3::new([[T::zero(); 3]; 3]);
            let mut row_nnz = 0;

            // Compute the dot product of the two sparse vectors.
            let mut row_iter = row.iter();
            let mut rhs_iter = rhs.as_data().iter();

            let mut col_mb = row_iter.next();
            let mut rhs_mb = rhs_iter.next();
            if col_mb.is_some() && rhs_mb.is_some() {
                loop {
                    if col_mb.is_none() || rhs_mb.is_none() {
                        break;
                    }
                    let (col_idx, col_block, _) = col_mb.unwrap();
                    let (rhs_idx, rhs_block, _) = rhs_mb.unwrap();

                    if rhs_idx < col_idx {
                        rhs_mb = rhs_iter.next();
                        continue;
                    } else if rhs_idx > col_idx {
                        col_mb = row_iter.next();
                        continue;
                    } else {
                        // rhs_idx == row_idx
                        sum_mtx += Matrix3::new(*rhs_block.into_arrays())
                            * Matrix3::new(*col_block.into_arrays()).transpose();
                        row_nnz += 1;
                        rhs_mb = rhs_iter.next();
                        col_mb = row_iter.next();
                    }
                }
            }

            if row_nnz > 0 {
                let (index, out_block) = out_iter_mut.next().unwrap();
                *index = row_idx;
                *out_block.into_arrays().as_mut_tensor() = sum_mtx;
                num_non_zeros += 1;
            }
        }

        num_non_zeros
    }
}

// 3x1 block matrix impls

impl<'a, T: Scalar, I: Set + AsIndexSlice> SSBlockMatrix3x1<T, I> {
    pub fn write_img<P: AsRef<std::path::Path>>(&'a self, path: P) {
        use image::ImageBuffer;

        let nrows = self.num_total_rows();
        let ncols = self.num_total_cols();
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
        let view = self.view().into_data();
        if let Ok(row) = view
            .selection
            .indices
            .binary_search(&(r / 3))
            .map(|idx| view.source.isolate(idx))
        {
            row.selection
                .indices
                .binary_search(&c)
                .map(|idx| row.source.isolate(idx).at(r % 3)[c])
                .unwrap_or_else(|_| T::zero())
        } else {
            T::zero()
        }
    }
}

impl<T: Scalar> SSBlockMatrix3x1<T> {
    /// Assume that rows are monotonically increasing in the iterator.
    pub fn from_block_triplets_iter_uncompressed<I>(
        iter: I,
        num_rows: usize,
        num_cols: usize,
    ) -> Self
    where
        I: Iterator<Item = (usize, usize, [T; 3])>,
    {
        let cap = iter.size_hint().0;
        let mut rows = Vec::with_capacity(cap);
        let mut cols = Vec::with_capacity(cap);
        let mut values: Vec<T> = Vec::with_capacity(cap);
        let mut offsets = Vec::with_capacity(num_rows);

        let mut prev_row = 0; // offset by +1 so we don't have to convert between isize.
        for (row, col, block) in iter {
            assert!(row + 1 >= prev_row); // We assume that rows are monotonically increasing.

            if row + 1 != prev_row {
                rows.push(row);
                prev_row = row + 1;
                offsets.push(cols.len());
            }

            cols.push(col);
            // Push each row at a time to produce a Vec<[T; 3]>
            values.push(block[0]);
            values.push(block[1]);
            values.push(block[2]);
        }
        offsets.push(cols.len());
        offsets.shrink_to_fit();
        rows.shrink_to_fit();

        let mut col_data = Chunked::from_offsets(
            offsets,
            Sparse::from_dim(
                cols,
                num_cols,
                Chunked3::from_flat(Chunked1::from_flat(values)),
            ),
        );

        col_data.sort_chunks_by_index();

        Sparse::from_dim(rows, num_rows, col_data).into_tensor()
    }
    pub fn from_block_triplets_iter<I>(iter: I, num_rows: usize, num_cols: usize) -> Self
    where
        I: Iterator<Item = (usize, usize, [T; 3])>,
    {
        Self::from_block_triplets_iter_uncompressed(iter, num_rows, num_cols).compressed()
    }
}

impl<T: Scalar, I: AsIndexSlice> SSBlockMatrix3x1<T, I>
where
    Self: for<'a> View<'a, Type = SSBlockMatrix3x1View<'a, T>>,
    I: IntoOwned<Owned = Vec<usize>>,
{
    /// Compress the matrix representation by consolidating duplicate entries.
    pub fn compressed(&self) -> SSBlockMatrix3x1<T> {
        let data = self.as_data();
        // Check that there are no duplicate rows. This should not happen when crating from
        // triplets.
        assert!(is_unique(data.selection.indices.as_ref()));
        Sparse::new(
            data.selection.view().into_owned(),
            data.view().source.compressed(|a, b| {
                *AsMutTensor::as_mut_tensor(a.as_mut_arrays()) += b.into_arrays().as_tensor()
            }),
        )
        .into_tensor()
    }
}

impl<T: Scalar, I: AsIndexSlice> SSBlockMatrix3x1<T, I>
where
    Self: for<'a> View<'a, Type = SSBlockMatrix3x1View<'a, T>>,
    I: IntoOwned<Owned = Vec<usize>>,
{
    /// Remove all elements that do not satisfy the given predicate and compress the resulting matrix.
    pub fn pruned(
        &self,
        keep: impl Fn(usize, usize, &Matrix3x1<T>) -> bool,
        mapping: impl FnMut(usize, usize),
    ) -> SSBlockMatrix3x1<T> {
        let data = self.as_data();
        // Check that there are no duplicate rows. This should not happen when crating from
        // triplets.
        assert!(is_unique(data.selection.indices.as_ref()));
        Sparse::new(
            data.selection.view().into_owned(),
            data.view().source.pruned(
                |a, b| *a.as_mut_arrays().as_mut_tensor() += b.into_arrays().as_tensor(),
                |i, j, e| keep(i, j, e.as_arrays().as_tensor()),
                mapping,
            ),
        )
        .into_tensor()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

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

    // #[test]
    // fn sparse_matrix_sparse_vector_mul() {
    //     // Variable length rows with one empty:
    //     // [2, 3]  [0 1]
    //     //         [0 0]
    //     let a = Sparse::from_dim(vec![0], 2, Chunked::from_sizes(vec![1], Sparse::from_dim(vec![1], 2, vec![1])));
    //     let b = Sparse::from_dim(vec![0,1], 2, vec![2,3]);
    //     let a_mtx = a.into_tensor().transpose();
    //     let b_vec = b.into_tensor();
    //     let out: Sparse<Vec<i32>> = a_mtx * b_vec;
    //     assert_eq!(Sparse::from_dim(vec![1], 2, vec![2]), out);
    // }

    #[test]
    fn block_sparse_matrix_sparse_vector_mul() {
        let a = Sparse::from_dim(
            vec![0],
            2,
            Chunked::from_sizes(
                vec![1],
                Sparse::from_dim(
                    vec![1],
                    2,
                    Chunked3::from_flat(Chunked3::from_flat(vec![1, 2, 3, 4, 5, 6, 7, 8, 9])),
                ),
            ),
        );
        let b = Sparse::from_dim(vec![0, 1], 2, Chunked3::from_flat(vec![1, 2, 3, 4, 5, 6]));
        let a_mtx = a.into_tensor();
        let b_vec = b.into_tensor();
        let out = a_mtx.view() * b_vec.view();
        assert_eq!(
            Sparse::from_dim(&[0][..], 2, Chunked3::from_flat(vec![32, 77, 122])),
            out.into_data()
        );
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
