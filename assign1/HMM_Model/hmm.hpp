#ifndef HMM_CLASS_HEADER  // for including this header file once
#define HMM_CLASS_HEADER
#define LOGZERO  (-1E10) // log(0)

#include <random> // The header for the generators.
#include <ctime> // To seed the generator.
#include "ndarray.hpp"
using namespace util;

namespace hmm {
    typedef unsigned int uint;
    class HMM {
    private:
        // training
        uint maxIter = 50;
        uint iter = 0;
        // model fact
        uint T;         // no. of time steps
        uint nStates;    // no. of states
        uint nObsTypes;  // no. of observations

        // model parameter
        array2d<double> logTrans;  // A matrix
        array2d<double> logEmis;   // B matrix
        array2d<double> logPi;   // pi
        vector<int> seq;      // observation sequence

        // another set of parameter for training and evaluating
        array2d<double> logAlpha;
        array2d<double> logBeta;
        array2d<double> logGamma;
        array3d<double> logDiGamma;

        // private functions
        void computeLogAlpha();
        void computeLogBeta();
        // void computeLogGamma();
        // void computeLogDiGamma();
        void computeLogDiGammaAndLogGamma();

        // update function call
        void updateLogPi();
        void updateLogEmis();
        void updateLogTrans();

        double getLogProbObsSeq() const;
        void normalizeModelParams();

        // merge param use
        double trainLogP;
        vector<double> logProbObsSeqTrained;  // <P(O_1|lammda_1), ....>

        // random number
        mt19937_64 rng;
        double randd();

    public:

        void makeModelParamGuess();
        void makeRandomGuess();
        void makeModelParamGuessUniform();
        void myModelGuess();
        void makeFlyPatternModelParamGuess();

        // constructor with guess
        HMM(const array2d<double>& trans, const array2d<double>& emis,
            const array2d<double>& pi);
        // constructor without guess
        HMM(uint nStates_, uint nObsTypes_, bool skipInit=false);
        ~HMM();

        // helper to print out model parameters
        void printOutModelParam(ostream& os=cout);
        void printOutObsSeq(ostream& os=cout);

        // observation seqence setter, pass by value
        void setObsSeq(const vector<int>&);

        // set model dimension
        void setModelDim(const uint nStates_, const uint nObsTypes);

        void train();

        void forecast(int& nextObsIdx, double& pForecast);
        void cheapForecast(int& curObsIdx, int& nextObsIdx, double& pForecast);

        double evaluateLogPObs();
        double evaluatePObs();

        void setMaxIteration(uint);

        void getModelParams(array2d<double>& trans, array2d<double>& emis,
                            array2d<double>& pi) const;
        uint getTrainIterations() const;

        void mergeParams(const array2d<double>& trans, const array2d<double>& emis,
                         const array2d<double>& pi, double pObsSeq);
        uint getNumTrainSeq() const;

        double getTrainProb() const;
        double getLogTrainProb() const;
        array2d<double> forecastVectorObs();

    }; // end class HMM

}; // end namespace hmm
#endif // UTIL_ARRAY3D_HEADER_