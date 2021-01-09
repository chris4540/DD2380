/*
 *  Uility matrix class header file
 */
#ifndef UTIL_MATRIX_HEADER_  // for including this header file once
#define UTIL_MATRIX_HEADER_
#include <iostream>

typedef unsigned int uint;

namespace util_matrix {

class matrix {
    private:
        uint m_;    // number of rows
        uint n_;    // number of cols
        double  *val_; // starting address of our value content.
        void swap(matrix&);  // method to swap us with input matrix

    public:
        // Default constructor
        matrix() : m_(0), n_(0), val_(nullptr) {};
        // Default destructor
        ~matrix();

        // Copy constructor, since our class has a ptr attribute.
        matrix(const matrix&);

        // Construct matrix with size only
        matrix(uint rows, uint cols);

        // Constructor with values
        matrix(uint rows, uint cols, const double *values);

        // Get the number of rows
        uint m() const;

        // Get the number of columns
        uint n() const;

        // Get the M_ij element
        double get(uint i, uint j) const;

        // Set the M_ij element
        void set(uint i, uint j, double val);

        // overload the assignment operators
        double& operator()(uint i, uint j);

        //
        double operator()(uint i, uint j) const;

        // Copy assignment
        matrix& operator=(const matrix&);

        // multiplication
        matrix matmul(const matrix&);


        friend std::ostream& operator<<(std::ostream& out, const matrix& a);
};  // end of class matrix definition
    // for output use
    std::ostream& operator<<(std::ostream&, const matrix&);

} // end namespace util_matrix
#endif // UTIL_MATRIX_HEADER_