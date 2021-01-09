#include <iostream>
#include <vector>
#include <assert.h>
#include "matrix.hpp"

using namespace std;
using namespace util_matrix;


int main(){

    matrix m1(10, 20);  // test if ok to declare

    vector<double> val2 {0., 1., 2., 3., 4., 5., 6., 7., 8., 9.};
    matrix m2(5, 2, val2.data());

    // test if ok to print out values

    // TODO: change it to assert string eq.
    for (uint i = 0; i < m2.m(); ++i){
        for (uint j = 0; j < m2.n(); ++j){
            cout << m2.get(i, j) << " ";
        }
    }
    cout << endl;

    //test if ok to assign value
    m2.set(0, 1, 99.0);
    assert(m2(0, 1) == 99.0);

    m2(1, 1) = 3.14;
    assert(m2(1, 1) == 3.14);

    // test if ok to operate through pointer
    double *x;
    x = &m2(4, 0);
    *x = 1000.0;
    assert(m2(4, 0) == 1000.0);

    // test if matmul
    matrix m3;
    try {
        m3 = m1.matmul(m2);
    } catch (std::length_error& e) {
        // cout << "Pass dimension checking!" << endl;
    }

    // ------------------------------------------------
    matrix a(2, 2);
    a(0, 0) = 1.0;
    a(0, 1) = 2.0;
    a(1, 0) = 3.0;
    a(1, 1) = 4.0;

    matrix b(2, 1);
    b(0, 0) = 5.0;
    b(1, 0) = 6.0;

    matrix c;
    c = a.matmul(b);

    // check size
    assert(c.m() == 2);
    assert(c.n() == 1);

    assert(c.get(0, 0) == 17);
    assert(c(1, 0) == 39);
    return 0;
}