#![cfg(feature = "sprs")]

//! A compatibility layer with the `sprs` sparse linear algebra library.

use super::*;
use std::convert::AsRef;

impl<S, I> Into<sprs::CsMat<f64>> for DSBlockMatrixBase<S, I, U3, U3>
where
    // Needed for num_cols/num_rows
    S: Set + IntoData,
    I: Set + IntoOwned<Owned = Vec<usize>> + AsRef<[usize]>,
    S::Data: IntoStorage,
    <S::Data as IntoStorage>::StorageType: IntoOwned<Owned = Vec<f64>>,
    // Needed for view
    Self: for<'a> View<'a, Type = DSBlockMatrix3View<'a>>,
{
    fn into(self) -> sprs::CsMat<f64> {
        let num_rows = self.view().num_total_rows();
        let num_cols = self.view().num_total_cols();

        let data = self.into_data();
        let Chunked {
            chunks: block_offsets,
            data:
                Sparse {
                    selection:
                        Select {
                            indices: block_indices,
                            ..
                        },
                    source,
                },
        } = data;
        let values = source.into_storage().into_owned();

        let mut offsets = block_offsets.into_inner().into_owned();
        offsets.iter_mut().for_each(|off| *off *= 3);
        let indices: Vec<usize> = block_indices
            .as_ref()
            .iter()
            .flat_map(|blk_idx| (0..3).map(move |i| 3 * blk_idx + i))
            .collect();

        sprs::CsMat::new((num_rows, num_cols), offsets, indices, values)
    }
}

impl Into<sprs::CsMat<f64>> for DSMatrix {
    fn into(self) -> sprs::CsMat<f64> {
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
