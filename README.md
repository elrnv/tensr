# `tensr`

[![Build status](https://travis-ci.org/elrnv/tensr.svg?branch=master)](https://travis-ci.org/elrnv/tensr)

A prototype for a linear algebra library that aims at supporting:
 - Multi-demensional array ("tensor") arithmetic (including scalars, vectors, matrices and higher dimensional structures),
 - Small (array based) tensors,
 - Dense dynamically allocated tensors,
 - Sparse tensors,
 - Lazy arithmetic,
 - Block (sparse or dense) matrices,
 - Special matrices (block diagonal, lower triangular, upper triangular, etc.),
 - Flat data layout in all tensor types for faster processing,
 - Arithmetic between compatible tensors.

Some subset of the features above has already been implemented.

The goals of this library (in no particular order) are
 - performance
 - simplicity (usage as well as implementation)

Meeting these goals is work in progress.

Higher level algorithms like decompositions are currently outside the scope of this project and are
expected to be provided by third-party crates.

# License

This repository is licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or https://www.apache.org/licenses/LICENSE-2.0)
 * MIT License ([LICENSE-MIT](LICENSE-MIT) or https://opensource.org/licenses/MIT)

at your option.
