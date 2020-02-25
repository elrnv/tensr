/// This suite of tests checks various tensor operations.
use tensr::*;

#[test]
fn tensor_subassign() {
    // Subtract a subset from a vec.
    let a = Subset::from_unique_ordered_indices(vec![1, 3, 4, 5], vec![1, 2, 3, 4, 5, 6, 7]);
    let mut tensor = Vector::new(vec![5, 6, 7, 8]);
    *&mut tensor.expr_mut() -= SubsetTensor::new(a.clone()).expr();
    assert_eq!(vec![3, 2, 2, 2], tensor.data);

    // Subtract subset view from a vec
    let mut tensor = Vector::new(vec![5, 6, 7, 8]);
    *&mut tensor.expr_mut() -= a.view().into_tensor().expr();
    assert_eq!(vec![3, 2, 2, 2], tensor.data);

    // Subtract subset view ref from a vec
    let mut tensor = Vector::new(vec![5, 6, 7, 8]);
    *&mut tensor.expr_mut() -= a.view().expr();
    assert_eq!(vec![3, 2, 2, 2], tensor.data);

    // Subtract subset ref from a vec
    let mut tensor = Vector::new(vec![5, 6, 7, 8]);
    *&mut tensor.expr_mut() -= a.expr();
    assert_eq!(vec![3, 2, 2, 2], tensor.data);
}
