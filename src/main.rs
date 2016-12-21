mod gsl {
    use std::ops::Drop;
    use std::fmt;

    #[repr(C)]
    enum CBLAS_TRANSPOSE {
        CblasNoTrans=111,
        CblasTrans=112,
        CblasConjTrans=113
    }

    #[repr(C)]
    struct CBlock {
        size: usize,
        data: *mut f64,
    }

    /*
    impl Copy for CBlock {}

    impl Clone for CBlock {
        fn clone(&self) -> CBlock {
            *self
        }
    }
     */

    #[repr(C)]
    pub struct CMatrix {
        size1: usize,
        size2: usize,
        tda: usize,
        data: *mut f64,
        block: *mut CBlock,
        owner: i32,
    }

    /*
    impl Copy for CMatrix {}

    impl Clone for CMatrix {
        fn clone(&self) -> CMatrix {
            *self
        }
    }
     */

    #[link(name = "gsl")]
    extern {
        fn gsl_matrix_calloc(row: usize, col: usize) -> *mut CMatrix;
        fn gsl_matrix_free(m: *mut CMatrix);
        fn gsl_matrix_get(m: *const CMatrix, i: usize, j: usize) -> f64;
        fn gsl_matrix_set(m: *mut CMatrix, i: usize, j: usize, x: f64);
        fn gsl_blas_dgemm(TransA: CBLAS_TRANSPOSE, TransB: CBLAS_TRANSPOSE,
                              alpha: f64, A: *const CMatrix, B: *const CMatrix,
                              beta: f64, C: *mut CMatrix);
    }

    pub struct Matrix {
        m: *mut CMatrix,
    }

    impl Matrix {
        pub fn new(row: usize, col: usize) -> Matrix {
            let m = unsafe { gsl_matrix_calloc(row, col) };
            Matrix{ m: m }
        }
    }

    impl Drop for Matrix {
        fn drop(&mut self) {
            unsafe { gsl_matrix_free((*self).m); };
        }
    }

    impl Matrix {
        pub fn get(&self, i: usize, j: usize) -> f64 {
            unsafe { gsl_matrix_get((*self).m, i, j) }
        }
    }

    impl Matrix {
        pub fn set(&mut self, i: usize, j: usize, x: f64) {
            unsafe { gsl_matrix_set((*self).m, i, j, x); };
        }
    }

    impl Matrix {
        pub fn dot(a: & Matrix, b: & Matrix) -> Matrix {
            let c = Matrix::new(a.row(), b.col());
            unsafe {
                gsl_blas_dgemm(
                    CBLAS_TRANSPOSE::CblasNoTrans,
                    CBLAS_TRANSPOSE::CblasNoTrans,
                    1.0, a.m, b.m, 0.0, c.m
                )
            }
            c
        }
    }

    impl Matrix {
        fn row(& self) -> usize {
            unsafe { (*(*self).m).size1 }
        }
    }

    impl Matrix {
        fn col(& self) -> usize {
            unsafe { (*(*self).m).size2 }
        }
    }

    impl fmt::Display for Matrix {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            let mut output = String::new();
            for i in 0..self.row() {
                for j in 0..self.col() {
                    output += &format!("{}", self.get(i, j));

                    if j < self.col() - 1 {
                        output += ", "
                    }
                }

                output += &format!("\n");
            }
            write!(f, "{}", output)
        }
    }
}

fn main() {
    let v1 = vec![
        vec![2.15, 1.10, 5.15],
        vec![1.25, 2.35, 7.55],
        vec![2.44, 3.55, 8.25],
    ];

    let v2 = vec![
        vec![10.21, 9.22],
        vec![12.15, 7.22],
        vec![1.25, 3.25],
    ];

    let mut m1 = gsl::Matrix::new(3, 3);
    let mut m2 = gsl::Matrix::new(3, 2);

    for i in 0..v1.len() {
        for j in 0..v1[0].len() {
            m1.set(i, j, v1[i][j]);
        }
    }

    for i in 0..v2.len() {
        for j in 0..v2[0].len() {
            m2.set(i, j, v2[i][j]);
        }
    }

    let mut m3 = gsl::Matrix::dot(&m1, &m2);

    println!("{}", m1);
    println!("{}", m2);
    println!("{}", m3);
}
