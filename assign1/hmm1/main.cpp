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

vector<int> getIntArrayFromLine(const string& line) {
    //
    // Args:
    //  line :

    stringstream ss(line);   // convert line to stringstream

    // get the size of matrix first
    int size;
    ss >> size;


    // construct the result
    vector<int> ret;
    string buf;                 // a buffer string
    while(ss >> buf){
        ret.push_back(stoi(buf));
    }

    return ret;
}

void getMatrixsFromStdin(
        matrix& transition,
        matrix& emission,
        matrix& initialState,
        vector<int>& obsSeq)
{
    // get matrixes from stdin

    // output:
    //      transition
    //      emission
    //      initialState
    //      obsSeq

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

    // observation seq
    getline(cin, line);
    obsSeq = getIntArrayFromLine(line);

}

void printOut(
        const matrix& transition,
        const matrix& emission,
        const matrix& initialState,
        const vector<int>& obsSeq)
{
    // helper function to print stuff out
    cout << transition << endl;
    cout << emission << endl;
    cout << initialState << endl;

    for (auto i : obsSeq) cout << i << ' ';
    cout << endl;

}

void updateAlpha(const matrix& trans,
                 const matrix& emis,
                 const uint& obsIdx,
                 matrix& alpha)
{
    matrix predict;

    predict = alpha.matmul(trans);

    for (uint i = 0; i < alpha.n(); ++i){
        alpha.set(0, i, emis(i, obsIdx) * predict(0, i));
    }
}

int main() {

    matrix tran;
    matrix emis;
    matrix pi;
    vector<int> obsSeq;

    getMatrixsFromStdin(tran, emis, pi, obsSeq);
    // printOut(tran, emis, pi, obsSeq);

    // initialize alpha
    // alpha just the join prob P({O}, X_i) => alpha should be like pi
    matrix alpha(pi.m(), pi.n());
    uint obsIdx;
    for (uint i = 0; i < alpha.n(); ++i){
        obsIdx = obsSeq[0];
        alpha.set(0, i, pi(0, i)*emis(i, obsIdx));
        // cout << alpha << endl;
    }

    // start to update alpha
    for (uint t = 1; t < obsSeq.size(); ++t) {
        obsIdx = obsSeq[t];
        updateAlpha(tran, emis, obsIdx, alpha);
        // cout << alpha << endl;
    }

    // sum alpha over stats
    double sum = 0.0;
    for (uint j = 0; j < alpha.n(); ++j){
        sum += alpha(0, j);
    }

    cout << sum << endl;
    return 0;
}
