/*
    Uility matrix class implementation
*/

#include <algorithm>
#include <stdexcept>
#include <iostream>
#include "matrix.hpp"


using namespace util_matrix;

matrix::matrix(const matrix& mat)
        : m_{mat.m_}, n_{mat.n_},
          val_{(m_==0||n_==0) ? nullptr : new double[m_*n_]}
{
    std::copy(mat.val_, mat.val_ + (m_*n_), val_);
}

matrix::matrix(uint rows, uint cols)
    : m_{rows}, n_{cols},
      val_{(m_==0||n_==0) ? nullptr : new double[m_*n_]}
{
    std::fill(val_, val_ + (m_*n_), 0);
}

matrix::matrix(uint rows, uint cols, const double *values):
    m_{rows}, n_{cols},
    val_{(m_==0||n_==0) ? nullptr : new double[m_*n_]}
{
    std::copy((const double *)values, (const double *)(values + (m_*n_)), val_);
}

matrix::~matrix() {
    // deallocate value array
    delete[] val_;
}

uint matrix::m() const {
    return m_;
}

uint matrix::n() const {
    return n_;
}

double matrix::get(uint i, uint j) const {
    if (i >= m_ || j >= n_) {
        throw std::out_of_range("Index out of range! Please check!");
    }
    return val_[i*n_ + j];
}

void matrix::set(uint i, uint j, double val) {
    if (i >= m_ || j >= n_) {
        throw std::out_of_range("Index out of range! Please check!");
    }
    val_[i*n_ + j] = val;
}

double& matrix::operator()(uint i, uint j) {
    if (i >= m_ || j >= n_) {
        throw std::out_of_range("Index out of range! Please check!");
    }
    return val_[i*n_ + j];
}

double matrix::operator()(uint i, uint j) const{
    return get(i, j);
};

matrix matrix::matmul(const matrix& b){
    matrix& a = *this; // obtain self-referencing

    if (a.n() != b.m()) {
        throw std::length_error("Unmatched size of matrix was inputted!");
    }

    matrix mat(a.m(), b.n());
    double sum;
    for (uint i = 0; i < a.m(); i++) {
        for (uint j = 0; j < b.n(); j++) {
            sum = 0.0;
            for (uint k = 0; k < a.n(); k++){
                sum += a(i, k) * b(k, j);
            }
            mat(i, j) = sum;
        }
    }
    return mat;
};

void matrix::swap(matrix& other) {
    std::swap(m_, other.m_);
    std::swap(n_, other.n_);
    std::swap(val_, other.val_);
}

matrix& matrix::operator=(const matrix& a) {
    matrix tmp(a);
    this->swap(tmp);
    return *this;
}

namespace util_matrix {

     std::ostream& operator<<(std::ostream& out, const matrix& a) {
        out << '[';
        for (uint i = 0; i < a.m(); ++i) {
            if (i != 0) out << ";  ";
            for (uint j = 0; j < a.n(); ++j) {
                if (j != 0) out << ',';
                out << ' ' << a(i, j);
            }
        }
        out << ' ' << ']';
        return out;
    }
}