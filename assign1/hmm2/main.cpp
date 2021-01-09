#include <iostream>
#include <sstream>
#include <string>
#include <cmath>
#include <vector>

#include "matrix.hpp"

#define MIN -1e9

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

void updateDelta(const matrix& trans,
                 const matrix& emis,
                 const uint& obsIdx,
                 const uint& tIdx,
                 matrix& delta,
                 matrix& deltaIdx)
{

    // initialize temporay matrix
    matrix tmp(trans.m(), trans.n());
    double prod;

    // follow the notation in 2.18
    for (uint j = 0; j < tmp.m(); ++j){
        for (uint i = 0; i < tmp.n(); ++i){
            prod = pow(10, (log10(trans(j, i)) + log10(delta(0, j)) + log10(emis(i, obsIdx))));
            tmp.set(i, j, prod);
        }
    }

    // update delta with the maximum value of each row of tmp matrix
    // no handling on duplicated max
    for (uint i = 0; i < tmp.m(); ++i){
        double max = MIN;
        uint idx = -1;
        for (uint j = 0; j < tmp.n(); ++j){
            if (tmp(i, j) > max) {
                max = tmp(i, j);
                idx = j;
            }
        }

        deltaIdx.set(i, tIdx-1, idx);  // minus one time index due to definition
        delta.set(0, i, max);
    }
}

vector<int> getStateSeq(const matrix& delta, const matrix& deltaIdx) {
    // get state sequence by backtracing

    vector<int> ret;

    // find the best guess of state at the last time step
    double max = MIN;
    uint stateIdx = -1;
    for (uint i = 0; i < delta.n(); ++i){
        if (delta(0, i) > max){
            max = delta(0, i);
            stateIdx = i;
        }
    }

    ret.insert(ret.begin(), stateIdx);

    for (int t = deltaIdx.n() - 1; t >= 0; --t){
        stateIdx = deltaIdx(stateIdx, t);
        ret.insert(ret.begin(), stateIdx);
    }

    return ret;
}

int main() {

    matrix tran;
    matrix emis;
    matrix pi;
    vector<int> obsSeq;

    getMatrixsFromStdin(tran, emis, pi, obsSeq);
    // printOut(tran, emis, pi, obsSeq);

    // initialize delta
    matrix delta(pi.m(), pi.n());
    uint obsIdx;
    for (uint i = 0; i < delta.n(); ++i){
        obsIdx = obsSeq[0];
        delta.set(0, i, pi(0, i)*emis(i, obsIdx));
    }

    // init deltaIdx
    // deltaIdx record the argmax index from t = 1 to T-1
    matrix deltaIdx(tran.m(), obsSeq.size() - 1);
    // start to update delta
    for (uint t = 1; t < obsSeq.size(); ++t) {
        obsIdx = obsSeq[t];
        updateDelta(tran, emis, obsIdx, t, delta, deltaIdx);
    }

    vector<int> stateSeq;
    stateSeq = getStateSeq(delta, deltaIdx);

    for (auto i : stateSeq) cout << i << ' ';
    cout << endl;
    return 0;
}
