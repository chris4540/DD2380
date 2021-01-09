#include <iostream>
#include <assert.h>
#include "ndarray.hpp"

using namespace std;
int main(int argc, char const *argv[])
{
    util::array3d<double> A(1, 2, 3);

    A(0, 0, 0) = 100.0;
    assert(A(0, 0, 0) == 100.0);

    util::array3d<double> B;

    // test copying
    B = A;
    assert(B(0, 0, 0) == 100.0);

    // test init with array
    double val[] = {1, 2, 3, 4, 5, 6};
    util::array3d<double> C(1, 2, 3);
    C.assign(val);

    for (uint i = 0; i < C.m(); ++i)
    for (uint j = 0; j < C.n(); ++j)
    for (uint k = 0; k < C.o(); ++k)
        cout << C(i, j, k) << " ";
    cout << endl;


    double valE[] = {1, 2, 3, 4, 5, 6, 7, 8};
    util::array3d<double> E(2, 2, 2);
    E.assign(valE);
    cout << E << endl;


    util::array2d<double> D(2, 3);
    D.assign(val);
    cout << D << endl;

    // --------------------------------------------------------
    // check dipulicating
    util::array2d<double> F;
    F = D;

    F(0, 0) = 0;
    assert(D(0, 0) == 1.0);
    assert(F(0, 0) == 0.0);
    return 0;
}
