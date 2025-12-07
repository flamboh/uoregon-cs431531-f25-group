// symmetric_tucker_hooi.cpp
// Code written by ChatGPT to approximate algorithm written in 
// Efficient Parallel Sparse Symmetric Tucker Decomposition for High-Order Tensors paper
#include <Eigen/Dense> //Library doesn't exist we will write the code to perform the Eigen value functions
#include <vector>
#include <array>
#include <random>
#include <cmath>
#include <cassert>
#include <iostream>
#include <cstdint>

// Provided elsewhere (you said these are prewritten)
Eigen::MatrixXd S3TTMc(const SparseSymTensor& X, const Eigen::MatrixXd& U);
// Returns Y_mat of shape (I x R^{N-1}) — that is, mode-1 matricization after applying U
// along all other modes. Exact layout must match what SVD expects.
// This function is the major bottleneck of the operation

Eigen::MatrixXd SVD_topR(const Eigen::MatrixXd& M, int R);
// Returns left singular vectors (I x R) or top-R eigenvectors as needed.


using Matrix = Eigen::MatrixXd;
using Index = std::size_t;
using Value = double;

// ---------------------------
// Sparse symmetric tensor (canonical COO-like)
// ---------------------------
// We store each nonzero as a vector<int> of length N with canonical (sorted) indices
// and a value. The tensor is symmetric: any permutation of indices refers to same value.
// ---------------------------
struct SparseSymTensor {
    Index N;                       // tensor order
    Index mode_size;               // each mode size (assumed equal for symmetric case)
    std::vector<std::vector<Index>> indices; // length = nnz, each is vector<Index>(N)
    std::vector<Value> values;              // length = nnz

    Index nnz() const { return values.size(); }
};

// ---------------------------
// Utility: integer power
// ---------------------------
static inline Index ipow(Index a, Index b) {
    Index r = 1;
    while (b--) r *= a;
    return r;
}

// ---------------------------
// Compute dense core G (size R^N) by projection:
// G_{p0..p_{N-1}} += sum_{(i0..i_{N-1}) in NZ} X_{i...} prod_{m=0..N-1} U(i_m, p_m)
// Because tensor is symmetric, we iterate over stored canonical nonzeros.
// ---------------------------
std::vector<Value> compute_core_from_sparse(const SparseSymTensor& X,
                                            const Matrix& U, // (I x R)
                                            int R)           // rank per mode (same for symmetric)
{
    const Index N = X.N;
    const Index I = X.mode_size;
    assert(U.rows() == (int)I && U.cols() == R);

    Index core_size = ipow((Index)R, N);
    std::vector<Value> G(core_size, 0.0);

    // Precompute strides for linear indexing of the core
    std::vector<Index> strides(N);
    strides[0] = 1;
    for (Index m = 1; m < N; ++m) strides[m] = strides[m-1] * (Index)R;

    // Temporary buffer to hold the row-values for each mode for a single nonzero
    // rowVals[m][r] = U( i_m, r )
    std::vector<std::vector<Value>> rowVals(N, std::vector<Value>(R));

    // Multi-index to iterate over R^N combos; we update G[idx] += v * prod(rowVals[m][pm])
    std::vector<int> multi(N, 0);

    for (Index nz = 0; nz < X.nnz(); ++nz) {
        const std::vector<Index>& idxs = X.indices[nz];
        Value xval = X.values[nz];

        // load row values
        for (Index m = 0; m < N; ++m) {
            Index row = idxs[m];
            for (int r = 0; r < R; ++r) rowVals[m][r] = U((int)row, r);
        }

        // iterate all combinations p0..p_{N-1}
        // naive nested loops via incrementing multi[]; this is fine if R^N small
        // also compute product incrementally for slight speedup: compute product fresh each combo
        std::fill(multi.begin(), multi.end(), 0);
        Index linear = 0;
        while (true) {
            // compute product
            Value prod = xval;
            for (Index m = 0; m < N; ++m) prod *= rowVals[m][ multi[m] ];

            G[linear] += prod;

            // increment multi-index
            Index d = 0;
            while (d < N) {
                multi[d] += 1;
                linear += strides[d];
                if (multi[d] < R) break;
                // overflow, reset this digit
                multi[d] = 0;
                linear -= strides[d] * R; // subtract the added block
                d += 1;
            }
            if (d == N) break;
        }
    }

    return G;
}

// ---------------------------
// Symmetric HOOI driver
// Assumes: S3TTMc(X, U) and SVD_topR(Y_mat, R) are implemented externally.
// ---------------------------
struct HOOIResult {
    Matrix U;                 // factor matrix (I x R)
    std::vector<Value> G;     // core tensor (R^N) in linearized order with strides as in compute_core_from_sparse
};

HOOIResult symmetric_hooi(const SparseSymTensor& X,
                         int R,
                         int maxIter = 50,
                         double tol = 1e-6)
{
    const Index I = X.mode_size;
    const Index N = X.N;
    assert(N >= 2); // order must be >= 2

    // Random init for U (I x R)
    std::mt19937_64 rng(1234567);
    std::normal_distribution<double> nd(0.0,1.0);
    Matrix U = Matrix::Zero((int)I, R);
    for (Index i=0;i<I;++i) for (int r=0;r<R;++r) U((int)i,r) = nd(rng);

    // optionally orthonormalize initial U (QR)
    Eigen::HouseholderQR<Matrix> qr(U);
    U = qr.householderQ() * Matrix::Identity((int)I, R);

    Matrix U_new = U;
    for (int iter = 0; iter < maxIter; ++iter) {
        // 1) Compute Y = X x_{-1} U  (S3TTMc) --> returns matricization Y_{(1)} of shape I x R^{N-1}
        Matrix Y = S3TTMc(X, U); // prewritten function per user's assumption
        // Y must be (I x R^{N-1}). If your S3TTMc returns a tensor, you must convert it to matricization.

        // 2) Compute SVD (or eigen) on Y to get updated U (left singular vectors)
        U_new = SVD_topR(Y, R); // prewritten: returns I x R

        // 3) Compute core G = X x_1 U_new^T x_2 U_new^T ... x_N U_new^T
        //    We compute by accumulating contributions from nonzeros (sparse-friendly).
        std::vector<Value> G = compute_core_from_sparse(X, U_new, R);

        // 4) convergence check on factor U
        double num = (U_new - U).norm();
        double den = U.norm();
        double rel = (den == 0.0) ? num : num / den;
        std::cout << "[HOOI] iter " << iter << " rel-change(U)=" << rel << "\n";
        U = U_new;
        if (rel < tol) {
            return HOOIResult{U, std::move(G)};
        }
    }

    // final core
    std::vector<Value> G_final = compute_core_from_sparse(X, U, R);
    return HOOIResult{U, std::move(G_final)};
}

// ---------------------------
// Example usage (main) — placeholder S3TTMc and SVD must exist
// ---------------------------
int main() {
    // Example: 3rd-order symmetric tensor with mode size 5 and a few nonzeros
    SparseSymTensor X;
    X.N = 3;
    X.mode_size = 5;
    X.indices = {
        {0,0,1}, {1,2,2}, {3,3,4} // canonical index tuples
    };
    X.values = {1.5, -2.0, 0.7};

    int R = 2;
    HOOIResult res = symmetric_hooi(X, R, 20, 1e-5);

    std::cout << "Result U shape: " << res.U.rows() << " x " << res.U.cols() << "\n";
    std::cout << "Core size: " << res.G.size() << "\n";
    return 0;
}
