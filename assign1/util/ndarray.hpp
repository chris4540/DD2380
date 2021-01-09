#ifndef UTIL_ARRAY3D_HEADER_  // for including this header file once
#define UTIL_ARRAY3D_HEADER_
#include <vector>
#include <iostream>
using namespace std;

typedef unsigned int uint;

namespace util {

    // array3d<...> class
    template <typename T>
    class array3d {
    private:
        uint dim1;
        uint dim2;
        uint dim3;
        std::vector<T> data;
    public:
        // constructor
        array3d(uint dim1=0, uint dim2=0, uint dim3=0, T const& fill=T()) :
            dim1(dim1), dim2(dim2), dim3(dim3), data(dim1*dim2*dim3, fill)
        {}

        // assign an array
        void assign(const T *val) {
            data.assign(val, val + dim1*dim2*dim3);
        }

        // return reference
        T& operator()(uint i, uint j, uint k) {
            if (i >= dim1 || j >= dim2 || k >= dim3) {
                throw std::out_of_range("Index out of range! Please check!");
            }
            return data[i*dim2*dim3 + j*dim3 + k];
        }

        // return const object reference
        const T& operator()(uint i, uint j, uint k) const {
            if (i >= dim1 || j >= dim2 || k >= dim3) {
                throw std::out_of_range("Index out of range! Please check!");
            }
            return data[i*dim2*dim3 + j*dim3 + k];
        }

        // some return just for convention
        uint m() const { return dim1;}
        uint n() const { return dim2;}
        uint o() const { return dim3;}

        // print
        template <class T_>
        friend std::ostream& operator<<(std::ostream& os, const array3d<T_>& arr);
    };

    // array2d<...> class
    template <typename T>
    class array2d {
    private:
        uint dim1;
        uint dim2;
        std::vector<T> data;
    public:
        // constructor
        array2d(uint dim1=0, uint dim2=0, T const& fill=T()) :
            dim1(dim1), dim2(dim2), data(dim1*dim2, fill)
        {}

        // assign an array
        void assign(const T *val) {
            data.assign(val, val + dim1*dim2);
        }

        // return reference
        T& operator()(uint i, uint j) {
            if (i >= dim1 || j >= dim2) {
                throw std::out_of_range("Index out of range! Please check!");
            }
            return data[i*dim2 + j];
        }

        // return const object reference
        const T& operator()(uint i, uint j) const {
            if (i >= dim1 || j >= dim2) {
                throw std::out_of_range("Index out of range! Please check!");
            }
            return data[i*dim2 + j];
        }

        // some return just for convention
        uint m() const { return dim1;}
        uint n() const { return dim2;}

        template <class T_>
        friend std::ostream& operator<<(std::ostream& os, const array2d<T_>& arr);
    }; // end class def.

    // print 3d
    template <typename T>
    std::ostream& operator<<(std::ostream& os, const array3d<T>& arr){
        os << "[";
        for (uint i = 0; i < arr.m(); ++i) {
            for (uint j = 0; j < arr.n(); ++j) {
                if (j == 0) os << "[";
                if (j != 0) os << ";";
                for (uint k = 0; k < arr.o(); ++k) {
                    if (k != 0) os << ',';
                    os << ' ' << arr(i, j, k);
                }
                if (j == arr.n() - 1) os << "]; ";
            }
        }
        os << ']';
        return os;
    }

    // print 2d
    template <typename T>
    std::ostream& operator<<(std::ostream& os, const array2d<T>& arr){
        os << '[';
        for (uint i = 0; i < arr.m(); ++i) {
            if (i != 0) os << ";  ";
            for (uint j = 0; j < arr.n(); ++j) {
                if (j != 0) os << ',';
                os << ' ' << arr(i, j);
            }
        }
        os << ']';

        return os;
    }
} // end namespace util
#endif // UTIL_ARRAY3D_HEADER_