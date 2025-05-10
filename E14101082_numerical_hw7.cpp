#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <algorithm>

using matrix = std::vector<std::vector<double>>;
using vector = std::vector<double>;

void print_vector(const vector& vec, const std::string& name) {
    std::cout << name << " = [ ";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << std::fixed << std::setprecision(6) << vec[i] << (i == vec.size() - 1 ? " " : ", ");
    }
    std::cout << "]" << std::endl;
}

vector subtract(const vector& a, const vector& b) {
    size_t n = a.size();
    vector res(n);
    for (size_t i = 0; i < n; ++i) {
        res[i] = a[i] - b[i];
    }
    return res;
}

vector add(const vector& a, const vector& b) {
    size_t n = a.size();
    vector res(n);
    for (size_t i = 0; i < n; ++i) {
        res[i] = a[i] + b[i];
    }
    return res;
}

vector scalar_multiply(double scalar, const vector& a) {
    size_t n = a.size();
    vector res(n);
    for (size_t i = 0; i < n; ++i) {
        res[i] = scalar * a[i];
    }
    return res;
}

double dot_product(const vector& a, const vector& b) {
    double res = 0.0;
    if (a.size() != b.size()) {
        std::cerr << "Error: Dot product dimension mismatch." << std::endl;
        return NAN;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        res += a[i] * b[i];
    }
    return res;
}

vector matrix_vector_multiply(const matrix& A, const vector& x) {
    size_t rows = A.size();
    if (rows == 0) return {};
    size_t cols = A[0].size();
    if (x.size() != cols) {
        std::cerr << "Error: Matrix-vector multiplication dimension mismatch." << std::endl;
        return {};
    }
    vector res(rows, 0.0);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            res[i] += A[i][j] * x[j];
        }
    }
    return res;
}

double infinity_norm(const vector& v) {
    double max_val = 0.0;
    if (v.empty()) return 0.0;
    for (double val : v) {
        if (std::abs(val) > max_val) {
            max_val = std::abs(val);
        }
    }
    return max_val;
}

double l2_norm(const vector& v) {
    if (v.empty()) return 0.0;
    double dp = dot_product(v,v);
    if (std::isnan(dp)) return NAN;
    return std::sqrt(dp);
}

vector jacobi(const matrix& A, const vector& b, vector x0, int max_iter, double tol_x_diff, int& iterations_taken) {
    int n = A.size();
    if (n == 0 || b.size() != n || x0.size() != n) {
        std::cerr << "Error: Jacobi input dimension mismatch." << std::endl;
        return {};
    }
    vector x = x0;
    vector x_new = x0;

    for (iterations_taken = 0; iterations_taken < max_iter; ++iterations_taken) {
        for (int i = 0; i < n; ++i) {
            double sigma = 0.0;
            for (int j = 0; j < n; ++j) {
                if (i != j) {
                    sigma += A[i][j] * x[j];
                }
            }
            if (std::abs(A[i][i]) < 1e-12) {
                std::cerr << "Error: Jacobi diagonal element A[" << i << "][" << i << "] is zero or too small." << std::endl;
                return {};
            }
            x_new[i] = (b[i] - sigma) / A[i][i];
        }

        vector diff = subtract(x_new, x);
        if (infinity_norm(diff) < tol_x_diff) {
            x = x_new;
            iterations_taken++;
            break;
        }
        x = x_new;
    }
    if (iterations_taken == max_iter) {
        std::cout << "Jacobi method: Max iterations reached. Norm of difference for last step: " << infinity_norm(subtract(x_new, x)) << std::endl;
    }
    return x;
}

vector gauss_seidel(const matrix& A, const vector& b, vector x0, int max_iter, double tol_x_diff, int& iterations_taken) {
    int n = A.size();
     if (n == 0 || b.size() != n || x0.size() != n) {
        std::cerr << "Error: Gauss-Seidel input dimension mismatch." << std::endl;
        return {};
    }
    vector x = x0;

    for (iterations_taken = 0; iterations_taken < max_iter; ++iterations_taken) {
        vector x_old_iteration = x;
        for (int i = 0; i < n; ++i) {
            double sigma1 = 0.0;
            for (int j = 0; j < i; ++j) {
                sigma1 += A[i][j] * x[j];
            }
            double sigma2 = 0.0;
            for (int j = i + 1; j < n; ++j) {
                sigma2 += A[i][j] * x_old_iteration[j];
            }
             if (std::abs(A[i][i]) < 1e-12) {
                std::cerr << "Error: Gauss-Seidel diagonal element A[" << i << "][" << i << "] is zero or too small." << std::endl;
                return {};
            }
            x[i] = (b[i] - sigma1 - sigma2) / A[i][i];
        }

        vector diff = subtract(x, x_old_iteration);
        if (infinity_norm(diff) < tol_x_diff) {
            iterations_taken++;
            break;
        }
    }
     if (iterations_taken == max_iter) {
        std::cout << "Gauss-Seidel method: Max iterations reached." << std::endl;
    }
    return x;
}

vector sor(const matrix& A, const vector& b, vector x0, double omega, int max_iter, double tol_x_diff, int& iterations_taken) {
    int n = A.size();
    if (n == 0 || b.size() != n || x0.size() != n) {
        std::cerr << "Error: SOR input dimension mismatch." << std::endl;
        return {};
    }
    vector x = x0;

    if (omega <= 0 || omega >= 2) {
        std::cout << "Warning: SOR omega value " << omega << " is outside the typical convergence range (0, 2)." << std::endl;
    }

    for (iterations_taken = 0; iterations_taken < max_iter; ++iterations_taken) {
        vector x_old_iteration = x;
        for (int i = 0; i < n; ++i) {
            double sigma1 = 0.0;
            for (int j = 0; j < i; ++j) {
                sigma1 += A[i][j] * x[j];
            }
            double sigma2 = 0.0;
            for (int j = i + 1; j < n; ++j) {
                sigma2 += A[i][j] * x_old_iteration[j];
            }
            if (std::abs(A[i][i]) < 1e-12) {
                std::cerr << "Error: SOR diagonal element A[" << i << "][" << i << "] is zero or too small." << std::endl;
                return {};
            }
            double term_in_parentheses = b[i] - sigma1 - sigma2;
            x[i] = (1.0 - omega) * x_old_iteration[i] + (omega / A[i][i]) * term_in_parentheses;
        }

        vector diff = subtract(x, x_old_iteration);
        if (infinity_norm(diff) < tol_x_diff) {
            iterations_taken++;
            break;
        }
    }
    if (iterations_taken == max_iter) {
        std::cout << "SOR method: Max iterations reached." << std::endl;
    }
    return x;
}

vector steepest_descent_pdf(const matrix& A, const vector& b, vector x0, int max_iter, double tol_residual_norm, int& iterations_taken) {
    int n = A.size();
    if (n == 0 || b.size() != n || x0.size() != n) {
        std::cerr << "Error: Steepest Descent input dimension mismatch." << std::endl;
        return {};
    }
    vector x = x0;
    vector r;

    for (iterations_taken = 0; iterations_taken < max_iter; ++iterations_taken) {
        vector Ax = matrix_vector_multiply(A, x);
        if (Ax.empty()) return {};
        r = subtract(b, Ax);

        double r_norm = l2_norm(r);
        if (std::isnan(r_norm)) return {};

        if (r_norm < tol_residual_norm) {
            iterations_taken++;
            break;
        }
        
        if (dot_product(r,r) < 1e-24) {
             iterations_taken++;
             break;
        }

        vector Ar = matrix_vector_multiply(A, r);
        if (Ar.empty()) return {};
        double r_dot_Ar = dot_product(r, Ar);
        if (std::isnan(r_dot_Ar)) return {};

        if (std::abs(r_dot_Ar) < 1e-20) {
            std::cout << "Steepest Descent: r_dot_Ar (" << r_dot_Ar <<") is close to zero. Algorithm cannot proceed effectively." << std::endl;
            iterations_taken++;
            break;
        }

        double alpha_k = dot_product(r,r) / r_dot_Ar;
        x = add(x, scalar_multiply(alpha_k, r));
    }
    if (iterations_taken == max_iter) {
        vector current_Ax = matrix_vector_multiply(A,x);
        if (!current_Ax.empty()){
            vector current_r = subtract(b, current_Ax);
            if (l2_norm(current_r) >= tol_residual_norm){
                 std::cout << "Steepest Descent (PDF version): Max iterations reached. Final residual norm: " << l2_norm(current_r) << std::endl;
            }
        } else {
            std::cout << "Steepest Descent (PDF version): Max iterations reached (error in final residual check)." << std::endl;
        }
    }
    return x;
}

int main() {
    matrix A = {
        { 4.0, -1.0,  0.0, -1.0,  0.0,  0.0},
        {-1.0,  4.0, -1.0,  0.0, -1.0,  0.0},
        { 0.0, -1.0,  4.0,  0.0,  1.0, -1.0},
        {-1.0,  0.0,  0.0,  4.0, -1.0, -1.0},
        { 0.0, -1.0,  0.0, -1.0,  4.0, -1.0},
        { 0.0,  0.0, -1.0,  0.0, -1.0,  4.0}
    };

    vector b_vec = {0.0, -1.0, 9.0, 4.0, 8.0, 6.0};
    int n_vars = b_vec.size();
    vector x0(n_vars, 0.0);

    double tol_x_diff = 1e-6;
    double tol_residual_norm = 1e-6;
    int max_iter = 1000; 

    int iterations_taken;

    std::cout << "Homework 7 Solutions:\n" << std::endl;
    std::cout << std::fixed << std::setprecision(8);

    iterations_taken = 0;
    vector x_jacobi = jacobi(A, b_vec, x0, max_iter, tol_x_diff, iterations_taken);
    std::cout << "(a) Jacobi Method:" << std::endl;
    if (!x_jacobi.empty()) {
        print_vector(x_jacobi, "x_jacobi");
        std::cout << "Iterations: " << iterations_taken << std::endl;
        vector Ax_jacobi = matrix_vector_multiply(A, x_jacobi);
        if (!Ax_jacobi.empty()){
            vector residual_jacobi = subtract(Ax_jacobi, b_vec);
            std::cout << "L2 Norm of Residual (Ax-b): " << l2_norm(residual_jacobi) << std::endl;
        }
    }
    std::cout << "\n---------------------------------\n" << std::endl;

    iterations_taken = 0;
    vector x_gauss_seidel = gauss_seidel(A, b_vec, x0, max_iter, tol_x_diff, iterations_taken);
    std::cout << "(b) Gauss-Seidel Method:" << std::endl;
    if (!x_gauss_seidel.empty()) {
        print_vector(x_gauss_seidel, "x_gauss_seidel");
        std::cout << "Iterations: " << iterations_taken << std::endl;
        vector Ax_gs = matrix_vector_multiply(A, x_gauss_seidel);
        if(!Ax_gs.empty()){
            vector residual_gs = subtract(Ax_gs, b_vec);
            std::cout << "L2 Norm of Residual (Ax-b): " << l2_norm(residual_gs) << std::endl;
        }
    }
    std::cout << "\n---------------------------------\n" << std::endl;

    iterations_taken = 0;
    double omega = 1.2; 
    vector x_sor = sor(A, b_vec, x0, omega, max_iter, tol_x_diff, iterations_taken);
    std::cout << "(c) SOR Method (omega = " << omega << "):" << std::endl;
    if (!x_sor.empty()) {
        print_vector(x_sor, "x_sor");
        std::cout << "Iterations: " << iterations_taken << std::endl;
        vector Ax_sor = matrix_vector_multiply(A, x_sor);
        if(!Ax_sor.empty()){
            vector residual_sor = subtract(Ax_sor, b_vec);
            std::cout << "L2 Norm of Residual (Ax-b): " << l2_norm(residual_sor) << std::endl;
        }
    }
    std::cout << "\n---------------------------------\n" << std::endl;

    iterations_taken = 0;
    vector x_steepest_descent = steepest_descent_pdf(A, b_vec, x0, max_iter, tol_residual_norm, iterations_taken);
    std::cout << "(d) Steepest Descent Method (PDF's 'Conjugate Gradient'):" << std::endl;
    if (!x_steepest_descent.empty()) {
        print_vector(x_steepest_descent, "x_steepest_descent");
        std::cout << "Iterations: " << iterations_taken << std::endl;
        vector Ax_sd = matrix_vector_multiply(A, x_steepest_descent);
        if(!Ax_sd.empty()){
            vector residual_sd = subtract(Ax_sd, b_vec);
            std::cout << "L2 Norm of Residual (Ax-b): " << l2_norm(residual_sd) << std::endl;
        }
    }
    std::cout << "\n---------------------------------\n" << std::endl;

    return 0;
}