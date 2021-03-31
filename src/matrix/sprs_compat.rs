#![cfg(feature = "sprs")]

//! A compatibility layer with the `sprs` sparse linear algebra library.

use super::*;
use std::convert::AsRef;

impl<S, I> Into<sprs::CsMat<f64>> for DSBlockMatrixBase<S, I, U3, U3>
where
    // Needed for num_cols/num_rows
    S: Set,
    I: Set + AsRef<[usize]>,
    // Needed for view
    Self: for<'a> View<'a, Type = DSBlockMatrix3View<'a>>,
{
    fn into(self) -> sprs::CsMat<f64> {
        let view = self.view();
        let num_rows = view.num_total_rows();
        let num_cols = view.num_total_cols();

        let view = view.as_data();
        let values = view.clone().into_storage().as_ref().to_vec();

        let (rows, cols) = {
            view.into_iter()
                .enumerate()
                .flat_map(move |(row_idx, row)| {
                    row.into_iter().flat_map(move |(col_idx, _)| {
                        (0..3).flat_map(move |row| {
                            (0..3).map(move |col| (3 * row_idx + row, 3 * col_idx + col))
                        })
                    })
                })
                .unzip()
        };

        sprs::TriMat::from_triplets((num_rows, num_cols), rows, cols, values).to_csr()
    }
}

impl<T> Into<sprs::CsMat<T>> for DSMatrix<T> {
    fn into(self) -> sprs::CsMat<T> {
        let num_rows = self.num_rows();
        let num_cols = self.num_cols();

        let Chunked {
            chunks: offsets,
            data:
                Sparse {
                    selection: Select { indices, .. },
                    source: values,
                },
        } = self.into_data();

        sprs::CsMat::new((num_rows, num_cols), offsets.into_inner(), indices, values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dsmatrix_sprs_conversion() {
        let vals = vec![1.0, 2.0, 3.0, 4.0];
        let rows = vec![0, 0, 1, 2];
        let cols = vec![0, 2, 0, 1];
        let iter = zip!(
            rows.iter().cloned(),
            cols.iter().cloned(),
            vals.iter().cloned()
        );
        let m = DSMatrix::from_triplets_iter(iter, 3, 3);
        let sprs_m: sprs::CsMat<f64> = m.clone().into();
        for (sprs_row, orig_row) in sprs_m.outer_iterator().zip(m.as_data().iter()) {
            for (sprs_val, orig_val) in sprs_row.iter().zip(orig_row.iter()) {
                assert_eq!(sprs_val, (orig_val.0, orig_val.1));
            }
        }
    }
}
