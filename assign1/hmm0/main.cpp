#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "matrix.hpp"

using namespace std;
using namespace util_matrix;

matrix getMatrixFromLine(const string& line) {

    stringstream ss(line);   // convert line to stringstream

    // get the size of matrix first
    int numRows, numCols;
    ss >> numRows >> numCols;

    // construct the result
    vector<double> content;
    string buf;                 // a buffer string
    while(ss >> buf){
        content.push_back(stod(buf));
    }

    matrix ret(numRows, numCols, content.data());
    return ret;
}

void getMatrixsFromStdin(matrix& transition, matrix& emission, matrix& initialState) {
    // get matrixes from stdin

    // output:
    //      transition
    //      emission
    //      initialState

    string line;
    // Transitional matrix
    getline(cin, line);
    transition = getMatrixFromLine(line);

    // emission
    getline(cin, line);
    emission = getMatrixFromLine(line);

    // initial state
    getline(cin, line);
    initialState = getMatrixFromLine(line);
}


int main() {

    matrix tran;
    matrix emis;
    matrix pi;

    getMatrixsFromStdin(tran, emis, pi);

    // Given the current state probability distribution what is the probabity
    // for the different emissions after the next transition
    // (i.e. after the system has made a single transition)?
    matrix result;
    result = pi.matmul(tran).matmul(emis);   // the emission prob. when t=1

    // return the dimension first
    uint numRows = result.m();
    uint numCols = result.n();
    cout << numRows << " " << numCols << " ";

    // return the values of the result matrix
    for(uint i = 0; i < numRows; i++){
        for(uint j = 0; j < numCols; j++){
            cout << result.get(i, j) << " ";
        }
    };

    cout << endl;
    return 0;
}
