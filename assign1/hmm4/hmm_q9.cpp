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
    for (uint n = 2; n < 6; n++) {
        hmm::HMM model(n, 4, true);
        model.makeRandomGuess();
        model.setObsSeq(obsSeq);
        model.setMaxIteration(100);
        model.train();
        cout << "Hidden States: " << n << endl;
        cout << model.getLogTrainProb() << endl;
    }
    cout << "=========================" << endl;

    vector<int> obs;
    for (uint i = 0; i < 300; ++i){
        obs.push_back(obsSeq[i]);
    }
    for (uint n = 2; n < 6; n++) {
        hmm::HMM model(n, 4, true);
        model.makeRandomGuess();
        model.setObsSeq(obs);
        model.setMaxIteration(100);
        cout << "Hidden States: " << n << endl;
        cout << model.getLogTrainProb() << endl;
        model.printOutModelParam();

   }
    {
        hmm::HMM model(tran, emis, pi);
        model.setObsSeq(obsSeq);
        double p = model.evaluatePObs();
        cout << "True P(O|lammda) " << p << endl;
    }
}
