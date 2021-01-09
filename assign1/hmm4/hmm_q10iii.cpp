#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "ndarray.hpp"
#include "hmm.hpp"

using namespace std;
using namespace util;

// ----------------------------------------------------------------------
// IO related
array2d<double> get2dArrayFromLine(const string& line) {

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

    array2d<double> ret(numRows, numCols);
    ret.assign(content.data());
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
        array2d<double>& transition,
        array2d<double>& emission,
        array2d<double>& initialState,
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
    transition = get2dArrayFromLine(line);

    // emission
    getline(cin, line);
    emission = get2dArrayFromLine(line);

    // initial state
    getline(cin, line);
    initialState = get2dArrayFromLine(line);

    // observation seq
    getline(cin, line);
    obsSeq = getIntArrayFromLine(line);

}
void printEncodeArray(const array2d<double>& a){
    cout << a.m() << " " << a.n() << " ";
    for (uint i = 0; i < a.m(); ++i)
    for (uint j = 0; j < a.n(); ++j)
        cout << a(i, j) << " ";
    cout << endl;
}

void printAns(const array2d<double>& trans, const array2d<double>& emis){
    printEncodeArray(trans);
    printEncodeArray(emis);
}
// End IO related
// -----------------------------------------------------------
int main()
{
    array2d<double> tran;
    array2d<double> emis;
    array2d<double> pi;
    vector<int> obsSeq;

    // read in A, B, pi
    getMatrixsFromStdin(tran, emis, pi, obsSeq);

    double tranVal[] ={
        0.65, 0.15, 0.2,
        0.11, 0.78, 0.11,
        0.19, 0.28, 0.53
    };
    tran.assign(tranVal);

    double emisVal[] ={
        0.65, 0.22, 0.08, 0.05,
        0.09, 0.38, 0.32, 0.21,
        0.01, 0.11, 0.19, 0.69
    };

    double piVal[] = {0.8, 0.1, 0.1};
    tran.assign(tranVal);
    emis.assign(emisVal);
    pi.assign(piVal);

    hmm::HMM model(tran, emis, pi);
    model.printOutModelParam();
    model.setObsSeq(obsSeq);
    model.setMaxIteration(100);
    cout << "Iter# " << model.getTrainIterations() << endl;
    cout << model.getLogTrainProb() << endl;
}