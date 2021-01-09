// HMM implementation file
#include <iostream>
#include <cassert>
#include <cmath>
#include <algorithm>
#include "hmm.hpp"

#define LOGBASE 10
#define LOGZERO  (-1E10) // log(0)
#define LOGTINY (-0.5E10) // log values < LOGTINY are set to LOGZERO
#define minLogExp -23

using namespace hmm;
using namespace std;

// ----------------------------------------------------------------------------
// helper functions
array2d<double> exp(const array2d<double>& mat){
    // take pow(base, <each 2dArray element>) and return

    array2d<double> ret(mat.m(), mat.n());
    for (uint i = 0; i < mat.m(); ++i)
        for (uint j = 0; j < mat.n(); ++j)
            ret(i, j) = pow(LOGBASE, mat(i, j));
    return ret;
}

array2d<double> log(const array2d<double>& mat){
    // take pow(base, <each 2dArray element>) and return

    array2d<double> ret(mat.m(), mat.n());
    for (uint i = 0; i < mat.m(); ++i) {
        for (uint j = 0; j < mat.n(); ++j) {

            if (mat(i, j) == 0) {
                ret(i, j) = LOGZERO;
            } else {
                ret(i, j) = log10(mat(i, j)) / log10(LOGBASE);
            }
        }
    }
    return ret;
}

double logAdd(double logP, double logQ) {
    // log(p + q) = log(p) + log(1 + q/p)
    double temp;
    double logRatio;
    double ratio;
    double ret;
    if (logP < logQ) {
        temp = logP;
        logP = logQ;
        logQ = temp;
    }
    // assert(logP >= logQ);
    logRatio = logQ - logP;

    // if q / p tends to 0, log(p + q) ~ log(p)
    if (logRatio < minLogExp) {
        if (logP > LOGTINY) {
            ret = logP;
        } else {
            ret = LOGZERO;
        }
    } else {
        ratio = pow(LOGBASE, logRatio);
        ret = logP + log10(1.0 + ratio) / log10(LOGBASE);
    }
    return ret;
}

array2d<double> normalizeLogProbMatrix(const array2d<double>& mat){
    //  apply constrain of row-stochastic matrix
    //  sum over each row equals one
    array2d<double> ret = mat; // copy first

    for (uint i = 0; i < mat.m(); ++i) {
        for (uint iter = 0; iter < 2; iter++){
            double logSum = LOGZERO;
            for (uint j = 0; j < mat.n(); ++j) {
                logSum = logAdd(ret(i, j), logSum);
            }

            double chk = LOGZERO;
            for (uint j = 0; j < mat.n(); ++j) {
                ret(i, j) = ret(i, j) - logSum;
                chk = logAdd(ret(i, j), chk);
            }
            if (abs(chk) < 1e-3) break;  // break iteration to sum to 1
        }
    }
    return ret;
};

array2d<double> normalizeProbMatrix(const array2d<double>& mat){
    //  apply constrain of row-stochastic matrix
    //  sum over each row equals one
    array2d<double> ret = mat; // copy first

    for (uint i = 0; i < mat.m(); ++i) {
        for (uint iter = 0; iter < 10; iter++){
            double sum = 0.0;
            for (uint j = 0; j < mat.n(); ++j) {
                sum += ret(i, j);
            }

            assert(sum > 0);
            for (uint j = 0; j < mat.n(); ++j) {
                ret(i, j) = ret(i, j) / sum;
            }
        }
    }
    return ret;
};
// ----------------------------------------------------------------------------

HMM::HMM(const array2d<double>& trans_,
         const array2d<double>& emis_,
         const array2d<double>& pi_) : rng((time(0)))
{
    // update the model parameter
    logTrans = log(trans_);
    logEmis = log(emis_);
    logPi = log(pi_);
    // update the model fact
    nStates = trans_.m();
    nObsTypes = emis_.n();
}

HMM::HMM(uint nStates_, uint nObsTypes_, bool skipInit): rng((time(0)))
{
    // copy the dimension info
    nStates = nStates_;
    nObsTypes = nObsTypes_;
    if (!skipInit) {
        makeModelParamGuessUniform();
    }
}

// default destructor
HMM::~HMM(){}

double HMM::randd() {
    // return a random number from 0 to 1
    // srand(time(NULL));
    // return (double)rand() / (RAND_MAX + 1.0);
    uniform_real_distribution<double> dis(0.0, 1.0);
    return dis(rng);
}

void HMM::myModelGuess(){
    double diaP = 0.7;
    double offDiaP = (1 - diaP) / (nStates - 1);
    array2d<double> trans(nStates, nStates, offDiaP);
    for (uint i = 0; i < nStates; ++i){
        trans(i, i) = diaP;
    }
    // add noise
    for (uint i = 0; i < nStates; ++i){
        for (uint j = 0; j < nStates; ++j) {
        trans(i, j) += (randd() - 0.5) / (randd() + 5);
        }
    }

    // Guess Emission prob. Avg guess
    // double emisP = 1.0 / nObsTypes;
    array2d<double> emis(nStates, nObsTypes, randd());

    // add noise
    for (uint i = 0; i < nStates; ++i) {
        for (uint k = 0; k < nObsTypes; ++k){
            emis(i, k) += (randd() - 0.5) / (randd() + 5) ;
        }
    }

    // Guess Pi
    // double valE[] = {1, 2, 3, 4, 5, 6, 7, 8};
    // double piP = 1.0 / nStates;
    array2d<double> pi(1, nStates, randd());
    // add noise
    for (uint i = 0; i < nStates; ++i) {
        pi(0, i) += (randd() - 0.5)/ (randd() + 5);
    }

    this->logTrans = log(normalizeProbMatrix(trans));
    this->logEmis = log(normalizeProbMatrix(emis));
    this->logPi = log(normalizeProbMatrix(pi));
}

void HMM::makeModelParamGuess(){

    assert(nStates > 1);

    // Guess Transition prob. as diagonal with litte off digonal terms
    double diaP = 0.6;
    double offDiaP = (1 - diaP) / (nStates - 1);
    array2d<double> trans(nStates, nStates, offDiaP);
    for (uint i = 0; i < nStates; ++i){
        trans(i, i) = diaP;
    }

    // add noise
    for (uint i = 0; i < nStates; ++i){
        for (uint j = 0; j < nStates; ++j) {
        trans(i, j) += (randd() - 0.5) / 10.0;
        }
    }

    // Guess Emission prob. Avg guess
    double emisP = 1.0 / nObsTypes;
    array2d<double> emis(nStates, nObsTypes, emisP);
    // add noise
    for (uint i = 0; i < nStates; ++i) {
        for (uint k = 0; k < nObsTypes; ++k){
            emis(i, k) += (randd() - 0.5) / 10.0 ;
        }
    }

    // Guess Pi
    double piP = 1.0 / nStates;
    array2d<double> pi(1, nStates, piP);
    // add noise
    for (uint i = 0; i < nStates; ++i) {
        pi(0, i) += (randd() - 0.5)/ 10.0;
    }

    this->logTrans = log(normalizeProbMatrix(trans));
    this->logEmis = log(normalizeProbMatrix(emis));
    this->logPi = log(normalizeProbMatrix(pi));
}

void HMM::makeFlyPatternModelParamGuess(){
    assert(nStates == 5);

    array2d<double> trans(nStates, nStates);
    // double valTrans[] = {0.60, .075, .075, .075, .075,   // Migrating
    //                      .075, 0.10, .075, .075, .075,   // Circling
    //                      .075, .075, 0.40, .075, .075,   // Hunting
    //                      .075, .075, .075, 0.50, .075,   // Drilling
    //                      .075, .075, .075, .075, 0.80};  // Zig-zag
    double valTrans[] = {0.232062, 0.000303248, 0.519423, 0.0245651, 0.223647,
                         0.000813303, 0.995806, 0.000577513, 0.00187922, 0.000923842,
                         0.190381, 0, 0.67501, 0.00739067, 0.127135,
                         0, 0.0256385, 0, 0.974361, 0,
                         0.30902, 0.000439186, 0.466502, 0.0530732, 0.170966};
    trans.assign(valTrans);
    // add noise
    for (uint i = 0; i < nStates; ++i){
        for (uint j = 0; j < nStates; ++j) {
        trans(i, j) += (randd() + 0.25) / 7.0;
        }
    }

    array2d<double> emis(nStates, nObsTypes, 0);
    // Migrating
    for (uint j = 0; j < nObsTypes; ++j)
        emis(0, j) = randd();
    // Circling: The bird gains height by flying up-left or up-right.
    emis(1, 0) = 0.5;
    emis(1, 2) = 0.5;
    // Hunting:
    emis(2, 3) = 0.5;
    emis(2, 5) = 0.5;
    // Drilling:
    for (uint j = 0; j < nObsTypes; ++j)
        emis(3, j) = randd() / 10;
    // Zig-zag:
    for (uint j = 0; j < nObsTypes; ++j)
        emis(4, j) = randd() / 10;

    // add noise
    for (uint i = 0; i < nStates; ++i){
        for (uint j = 0; j < nObsTypes; ++j) {
        emis(i, j) += (randd() + 0.25) / 7.0;
        }
    }

    array2d<double> pi(1, nStates, 0);
    pi(0, 0) = 1.0;
    // add noise
    for (uint i = 0; i < nStates; ++i) {
        pi(0, i) += (randd() + 0.5)/ 100.0;
    }

    this->logTrans = log(normalizeProbMatrix(trans));
    this->logEmis = log(normalizeProbMatrix(emis));
    this->logPi = log(normalizeProbMatrix(pi));
}


void HMM::makeRandomGuess(){
    // A
    array2d<double> trans(nStates, nStates);
    for (uint i = 0; i < nStates; ++i) {
        for (uint j = 0; j < nStates; ++j) {
            trans(i, j) = randd();
        }
    }

    array2d<double> emis(nStates, nObsTypes);
    for (uint i = 0; i < nStates; ++i) {
        for (uint k = 0; k < nObsTypes; ++k){
            emis(i, k) =  randd();
        }
    }

    array2d<double> pi(1, nStates);
    for (uint i = 0; i < nStates; ++i) {
        pi(0, i) =  randd();
    }

    this->logTrans = log(normalizeProbMatrix(trans));
    this->logEmis = log(normalizeProbMatrix(emis));
    this->logPi = log(normalizeProbMatrix(pi));
}

void HMM::makeModelParamGuessUniform(){
    // fix this, it will goes to local minimum

    assert(nStates >= 2);

    // Guess Transition prob. as diagonal with litte off digonal terms
    array2d<double> trans(nStates, nStates, 1.0 / nStates);
    for (uint i = 0; i < nStates; ++i){
        for (uint j = 0; j < nStates; ++j) {
        trans(i, i) += (randd() - 0.5) / 10.0;
        }
    }

    // Guess Emission prob. Avg guess
    double emisP = 1.0 / nObsTypes;
    array2d<double> emis(nStates, nObsTypes, emisP);
    for (uint i = 0; i < nStates; ++i) {
        for (uint k = 0; k < nObsTypes; ++k){
            emis(i, k) += (randd() - 0.5) / 10.0 ;
        }
    }
    // Guess Pi
    double piP = 1.0 / nStates;
    array2d<double> pi(1, nStates, piP);
    for (uint i = 0; i < nStates; ++i) {
        pi(0, i) += (randd() - 0.5)/ 10.0;
    }

    this->logTrans = log(normalizeProbMatrix(trans));
    this->logEmis = log(normalizeProbMatrix(emis));
    this->logPi = log(normalizeProbMatrix(pi));
}


void HMM::printOutModelParam(ostream& os){
    os << "A: " << exp(logTrans) << endl;
    os << "B: " << exp(logEmis) << endl;
    os << "pi: " << exp(logPi) << endl;
}

void HMM::printOutObsSeq(ostream& os) {
    for (auto i : seq) os << i << ' ';
    os << endl;
}

void HMM::setObsSeq(const vector<int>& seq_){
    // update the observation seq
    seq = seq_;
    // upate the fact
    T = seq.size();
}

void HMM::train(){
    double lastLogP = LOGZERO;
    for (iter = 0; iter < maxIter; ++iter){
        computeLogAlpha();
        computeLogBeta();
        computeLogDiGammaAndLogGamma();
        // computeLogGamma();

        updateLogPi();
        updateLogEmis();
        updateLogTrans();

        // check the prob of obs seq
        double logP = getLogProbObsSeq();
        if (logP > lastLogP) {
            // update prob. and keep iteration
            lastLogP = logP;
        } else {
            // stop iteration
            break;
        }
    }
    // if (iter % 10 != 0){
    normalizeModelParams();
    // }
    // save trainLogP
    trainLogP = lastLogP;
    // logProbObsSeqTrained.assign(1, lastLogP);
};

void HMM::normalizeModelParams(){
    logTrans = normalizeLogProbMatrix(logTrans);
    logEmis = normalizeLogProbMatrix(logEmis);
    logPi = normalizeLogProbMatrix(logPi);
}

void HMM::forecast(int& nextObsIdx, double& pForecast){
    // give out the next most possible observation
    // guess_pi
    // formula:
    // double normalizer = LOGZERO;
    // for (uint i = 0; i < nStates; ++i) {
    //     normalizer = logAdd(logEmis(i, lastObsIdx), normalizer);
    // }
    computeLogAlpha();

    double logPOs = getLogProbObsSeq();
    array2d<double> gLogPi(1, nStates);  // guess log pi
    for (uint i = 0; i < nStates; ++i)
        gLogPi(0, i) = logAlpha(T-1, i) - logPOs;


    array2d<double> LogObsProb(1, nObsTypes, LOGZERO);

    for (uint i = 0; i < nStates; ++i)
    for (uint j = 0; j < nStates; ++j)
    for (uint k = 0; k < nObsTypes; ++k)
        LogObsProb(0, k) = gLogPi(0, i) + logTrans(i, j) + logEmis(j, k);

    double maxLogP = LOGZERO;
    for (uint k = 0; k < nObsTypes; ++k){
        double logP = LogObsProb(0, k);
        if (logP > maxLogP){
            maxLogP = logP;
            nextObsIdx = k;
        }
    }
    pForecast = pow(LOGBASE, maxLogP);
}

array2d<double> HMM::forecastVectorObs(){
    // return pi_cap* A * b
    // please call this right after train!

    // computeLogAlpha();

    double logPOs = getLogProbObsSeq();
    array2d<double> gLogPi(1, nStates);  // guess log pi
    for (uint i = 0; i < nStates; ++i)
        gLogPi(0, i) = logAlpha(T-1, i) - logPOs;


    array2d<double> LogObsProb(1, nObsTypes, LOGZERO);
    for (uint i = 0; i < nStates; ++i)
    for (uint j = 0; j < nStates; ++j)
    for (uint k = 0; k < nObsTypes; ++k)
        LogObsProb(0, k) = gLogPi(0, i) + logTrans(i, j) + logEmis(j, k);

    return exp(normalizeLogProbMatrix(LogObsProb));
}

void HMM::cheapForecast(int& curObsIdx, int& nextObsIdx, double& pForecast){
    array2d<double> gLogPi(1, nStates);  // guess log pi
    for (uint i = 0; i < nStates; ++i)
        gLogPi(0, i) = logEmis(i, curObsIdx);

    gLogPi = normalizeLogProbMatrix(gLogPi);

    array2d<double> LogObsProb(1, nObsTypes, LOGZERO);

    for (uint i = 0; i < nStates; ++i)
    for (uint j = 0; j < nStates; ++j)
    for (uint k = 0; k < nObsTypes; ++k)
        LogObsProb(0, k) = gLogPi(0, i) + logTrans(i, j) + logEmis(j, k);

    // find max
    double maxLogP = LOGZERO;
    for (uint k = 0; k < nObsTypes; ++k){
        double logP = LogObsProb(0, k);
        if (logP > maxLogP){
            maxLogP = logP;
            nextObsIdx = k;
        }
    }
    pForecast = pow(LOGBASE, maxLogP);

}


void HMM::computeLogAlpha() {
    // compute alpha with forward algorithm.
    // compute from t = 0 to t = T - 1
    // symbol def.
    //  logAlpha: logAlpha(time_idx, state_idx)

    array2d<double> logAlpha(T, nStates); // all elements are taken log10

    // take care of t = 0
    for (uint i = 0; i < nStates; ++i)
        logAlpha(0, i) = logPi(0, i) + logEmis(i, seq[0]);

    // for t > 0
    for (uint t = 1; t < T; ++t) {
        for (uint i = 0; i < nStates; ++i) {
            double logP = LOGZERO;
            for(uint j = 0; j < nStates; ++j){
                logP = logAdd(logP, logTrans(j, i) + logAlpha(t-1, j));
            }
            logAlpha(t, i) = logEmis(i, seq[t]) + logP;
        }
    }
    this->logAlpha = logAlpha;
}

void HMM::computeLogBeta() {
    // compute beta with backward algorithm.
    // compute from t = T -1 to t = 0
    // symbol def.
    //  LogBeta: LogBeta(time_idx, state_idx)

    array2d<double> logBeta(T, nStates); // all elements are taken log10

    // take care of t = T-1
    for (uint i = 0; i < nStates; ++i)
        logBeta(T-1, i) = 0;  // log(1) = 0;

    // for t = T-2 .. 0
    for (int t = T-2; t >= 0; --t) {
        for (uint i = 0; i < nStates; ++i) {
            double logP = LOGZERO;
            for(uint j = 0; j < nStates; ++j){
                // TODO: double check this statement
                logP = logAdd(
                    logBeta(t+1, j) + logEmis(j, seq[t+1]) + logTrans(i, j),
                    logP);
            }
            logBeta(t, i) = logP;
        }
    }
    this->logBeta = logBeta;
}

void HMM::computeLogDiGammaAndLogGamma() {
    // compute digamma
    // compute from t = 0 to t = T -1 to
    // symbol def.
    //  logDiGamma: logDiGamma(time_idx, state_idx_i, state_idx_j)

    array3d<double> logDiGamma(T, nStates, nStates);
    array2d<double> logGamma(T, nStates);
    /*
      calculate the dominator first
    */
    // log of the prob. of seeing this observation seq
    double logPObsSeq = LOGZERO;
    for (uint i = 0; i < nStates; ++i)
        logPObsSeq = logAdd(logPObsSeq, logAlpha(T-1, i));
    // assert(logPObsSeq > LOGZERO);
    // assert(logPObsSeq <= 0);

    for (uint t = 0; t < T-1; ++t){
        for (uint i = 0; i < nStates; ++i) {
            double logSum = LOGZERO;
            for (uint j = 0; j < nStates; ++j) {
                logDiGamma(t, i, j) = logAlpha(t, i) + logTrans(i, j)
                                    + logEmis(j, seq[t+1]) + logBeta(t+1, j)
                                    - logPObsSeq;
                logSum = logAdd(logDiGamma(t, i, j), logSum);
            }
            logGamma(t, i) = logSum;
        }
    }
    this->logDiGamma = logDiGamma;
    this->logGamma = logGamma;
}

// void HMM::computeLogGamma() {
//     // compute gamma by marginalizing digamma
//     // symbol def.
//     //  logGamma: logGamma(time_idx, state_idx)

//     array2d<double> logGamma(T, nStates);

//     for (uint t = 0; t < T; ++t) {
//         for (uint i = 0; i < nStates; ++i) {
//             double logSum = LOGZERO;
//             for (uint j = 0; j < nStates; ++j) {
//                 logSum = logAdd(logDiGamma(t, i, j), logSum);
//             }
//             logGamma(t, i) = logSum;
//         }
//     }
//     this->logGamma = logGamma;
// }

void HMM::updateLogPi(){
    // array2d<double> logPi(1, nStates);

    for (uint i = 0; i < nStates; ++i)
        logPi(0, i) = logGamma(0, i);

    // this->logPi = logPi;
};

void HMM::updateLogEmis() {
    // array2d<double> logEmis(nStates, nObsTypes);

    for (uint i = 0; i < nStates; ++i) {
        vector<double> logPObs(nObsTypes, LOGZERO);
        double logP2 = LOGZERO;
        for (uint t = 0; t < T-1; ++t) {
            logPObs[seq[t]] = logAdd(logPObs[seq[t]], logGamma(t, i));
            logP2 = logAdd(logP2, logGamma(t, i));
        }

        for (uint k = 0; k < nObsTypes; ++k)
            logEmis(i, k) = logPObs[k] - logP2;
    };
    // this->logEmis = logEmis;
}

void HMM::updateLogTrans() {
    // array2d<double> logTrans(nStates, nStates);

    for (uint i = 0; i < nStates; ++i) {
        // compute dominator
        double logP2 = LOGZERO;
        for (uint t = 0; t < T-1; ++t){
            logP2 = logAdd(logP2, logGamma(t, i));
        }

        // assert(logP2 > LOGZERO);

        for (uint j = 0; j < nStates; ++j) {
            // compute numerator
            double logP1 = LOGZERO;
            for (uint t = 0; t < T-1; ++t) {
                logP1 = logAdd(logDiGamma(t, i, j), logP1);
            }

            logTrans(i, j) = logP1 - logP2;
        } // end do j
    } // end do i
    // this->logTrans = logTrans;
}

double HMM::getLogProbObsSeq() const{
    // get the prob. of obs seq given the model param with alpha pass
    double ret = LOGZERO;
    for (uint i = 0; i < nStates; ++i){
        ret = logAdd(logAlpha(T-1, i), ret);
    }
    return ret;
}

void HMM::getModelParams(
    array2d<double>& trans,
    array2d<double>& emis,
    array2d<double>& pi) const {
    trans = exp(logTrans);
    emis = exp(logEmis);
    pi = exp(logPi);
}

double HMM::evaluateLogPObs() {
    // calculate Log(P(O|lammda))
    computeLogAlpha();
    double logP = getLogProbObsSeq();
    return logP;
}

double HMM::evaluatePObs() {
    return pow(LOGBASE, evaluateLogPObs());
}

uint HMM::getTrainIterations() const{
    return iter;
}

void HMM::setMaxIteration(uint maxIter_) {
    this->maxIter = maxIter_;
}

// void HMM::mergeParams(
//         const array2d<double>& trans_, const array2d<double>& emis_,
//         const array2d<double>& pi_, double logPObsSeq)
// {

//     assert(trans_.m() == nStates);
//     assert(emis_.n() == nObsTypes);

//     double w = 0;  // the weighting to current parameters
//     for (double logP : logProbObsSeqTrained) {
//         w += logP;
//     }

//     array2d<double> inputLogTrans = log(trans_);
//     array2d<double> inputLogEmis = log(emis_);
//     array2d<double> inputLogPi = log(pi_);

//     // A
//     array2d<double> mergedLogTrans(nStates, nStates, LOGZERO);
//     for (uint i = 0; i < nStates; ++i) {
//         for (uint j = 0; j < nStates; ++j) {
//             mergedLogTrans(i, j) =
//                 logAdd((w + logTrans(i, j)), (logPObsSeq + inputLogTrans(i, j)))
//                 - w - logPObsSeq;
//         }
//     }

//     // B
//     array2d<double> mergedLogEmis(nStates, nObsTypes, LOGZERO);
//     for (uint i = 0; i < nStates; ++i) {
//         for (uint k = 0; k < nObsTypes; ++k){
//             mergedLogEmis(i, k) =
//                 logAdd((w + logEmis(i, k)), (logPObsSeq + inputLogEmis(i, k)))
//                 - w - logPObsSeq;
//         }
//     }

//     // pi
//     array2d<double> mergedLogPi(1, nStates, LOGZERO);
//     for (uint i = 0; i < nStates; ++i) {
//         mergedLogPi(0, i) =
//             logAdd((w + logPi(0, i)), (logPObsSeq + inputLogPi(0, i)))
//             - w - logPObsSeq;
//     }


//     // merging to existing model
//     this->logTrans = normalizeLogProbMatrix(mergedLogTrans);
//     this->logEmis = normalizeLogProbMatrix(mergedLogEmis);
//     this->logPi = normalizeLogProbMatrix(mergedLogPi);

//     logProbObsSeqTrained.push_back(logPObsSeq);
// }

// uint HMM::getNumTrainSeq() const {
//     return logProbObsSeqTrained.size();
// }

double HMM::getTrainProb() const {
    return pow(LOGBASE, trainLogP);
}
double HMM::getLogTrainProb() const {
    return trainLogP;
}