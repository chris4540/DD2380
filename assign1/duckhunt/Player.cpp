#include "Player.hpp"
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <cmath>
#include "Constants.hpp"

// #define minProbSpec 0.7
#define FLYPATTERNS 5
#define ITER_SPECIES 20
#define ITER_MS 4
#define EMSEMBLE 1
#define partition 80

using namespace hmm;
using namespace std;

void normalizeVector(vector<double>& v){
    double sum = 0.0;

    for (uint i = 0; i < v.size(); ++i){
        sum += v[i];
    }

    for (uint i = 0; i < v.size(); ++i){
        v[i] = v[i] / sum;
    }
}

namespace ducks
{
// constructor!!
Player::Player(): speciesModels((uint)COUNT_SPECIES, vector<HMM*>()) {
    nShoot = 0;
    nHit = 0;

    totalShoot = 0;
    totalHit = 0;
};

Action Player::shoot(const GameState &pState, const Deadline &pDue)
{
    double nObs = 5;
    // cerr << "Groud: " << pState.getRound() << endl;
    // if (pState.getRound() < 8) { return cDontShoot;}
    // cerr << "Groud: " << pState.getRound() << endl;
    if (pState.getRound() < 1) { return cDontShoot;}
    if (pState.getBird(0).getSeqLength() < partition) { return cDontShoot;} // for guess spec
    if (nShoot >= 100) {return cDontShoot;}

    uint nBirds = pState.getNumBirds();
    int bestMvIdx = MOVE_DEAD;
    double pBestMv = .985;
    // double pBestMv = atof(getenv("Prob"));
    int targetBird = -1;

    //
    vector<HMM*> birdModels(nBirds, NULL);
    // check which bird fit the best with limit obs ( < 20)
    double pLogBestFit = LOGZERO;
    for (uint b = 0; b < nBirds; ++b){
        Bird bird = pState.getBird(b);
        if (bird.isDead()) {continue;}
        if (guessBirdSpecs(bird) == SPECIES_BLACK_STORK) {continue;}

        HMM* hmm;
        hmm = new HMM(1, COUNT_MOVE, true); // skip init model params
        (*hmm).makeRandomGuess();
        (*hmm).setMaxIteration(50);
        uint end = bird.getSeqLength();
        uint start = bird.getSeqLength() - nObs;
        vector<int> seq = getBirdMovementSeq(bird, start, end);
        // for (uint i = start; i < end; ++i){
        //     seq.push_back((int)bird.getObservation(i));
        // }
        (*hmm).setObsSeq(seq);
        (*hmm).train();
        // save model
        birdModels[b] = hmm;

        double pLogObs = (*hmm).getLogTrainProb();
        if (pLogObs > pLogBestFit) {
            pLogBestFit = (*hmm).getLogTrainProb();
            targetBird = b;
        }
    }
    if (targetBird == -1) {
        return cDontShoot;
    }

    // forecast
    HMM* hmm = birdModels[targetBird];
    array2d<double> pVec = (*hmm).forecastVectorObs();

    // find the max among all movement
    for (uint j = 0; j < pVec.n(); ++j){
        double e = pVec(0, j);
        if (e > pBestMv){
            pBestMv = e;
            bestMvIdx = j;
        }
    }
    if (bestMvIdx != MOVE_DEAD) {
        nShoot++;
        usedObs += nObs;
        return Action(targetBird, static_cast<EMovement>(bestMvIdx));
    }

    return cDontShoot;

}

std::vector<ESpecies> Player::guess(const GameState &pState, const Deadline &pDue)
{
    /*
     * Here you should write your clever algorithms to guess the species of each bird.
     * This skeleton makes no guesses, better safe than sorry!
     */
    totalShoot += nShoot;
    totalHit += nHit;
    cerr << "Hit: " << totalHit << " Shoot: " << totalShoot << endl;

    nShoot = 0;
    nHit = 0;

    std::vector<ESpecies> lGuesses;
    if (pState.getRound() == 0){
        lGuesses.assign(pState.getNumBirds(), SPECIES_PIGEON);
        return lGuesses;
    }

    for(uint i = 0; i < pState.getNumBirds(); ++i){
        Bird bird = pState.getBird(i);
        ESpecies sp = guessBirdSpecs(bird);
        lGuesses.push_back(sp);
    }
    cerr << endl;
    return lGuesses;
}

void Player::hit(const GameState &pState, int pBird, const Deadline &pDue)
{
    /*
     * If you hit the bird you are trying to shoot, you will be notified through this function.
     */
    nHit += 1;
    std::cerr << "HIT BIRD!!!" << std::endl;
}

void Player::reveal(const GameState &pState, const std::vector<ESpecies> &pSpecies, const Deadline &pDue)
{
    /*
     * If you made any guesses, you will find out the true species of those birds in this function.
     */
    // loop over species
    for (uint i = 0; i < pSpecies.size() && pDue.remainingMs() > 100; ++i){
    // for (uint i = 0; i < pSpecies.size(); ++i){
        Bird bird = pState.getBird(i);
        ESpecies sp = pSpecies[i];
        vector<int> seq = getBirdMovementSeq(bird, 0, partition);

        for (uint e = 0; e < EMSEMBLE; ++e) {
            HMM* birdHmm = new HMM(FLYPATTERNS, COUNT_MOVE);
            (*birdHmm).setMaxIteration(ITER_SPECIES);
            (*birdHmm).setObsSeq(seq);
            (*birdHmm).train();
            speciesModels[(int)sp].push_back(birdHmm);
        }
    }
    cerr << endl;
}

std::vector<int> Player::getBirdMovementSeq(const Bird& bird, uint start, uint end){
    std::vector<int> ret;
    for (uint i = start; i < (uint)bird.getSeqLength() && i < end; ++i){
        EMovement val = bird.getObservation(i);
        if (val == MOVE_DEAD) break;
        ret.push_back((int)val);
    }
    return ret;
}

ESpecies Player::guessBirdSpecs(const Bird& bird){

    vector<double> maxPAmgSpec((uint)COUNT_SPECIES, 0.0);
    ESpecies possSpec;
    double best_p_amg_sp = 0;
    for (uint s = 0; s < (uint)COUNT_SPECIES; ++s){
        vector<HMM*> thisSpecieModels = speciesModels[s];
        double best_p = 0;  // best among models
        for (HMM* hmm: thisSpecieModels){
            (*hmm).setObsSeq(getBirdMovementSeq(bird, 0, partition));
            double p = (*hmm).evaluatePObs();  // TODO

            if (p > best_p) {
                best_p = p;
            }
        }
        maxPAmgSpec[s] = best_p;
    }

    normalizeVector(maxPAmgSpec);

    for(uint i  = 0; i < maxPAmgSpec.size(); ++i){
        if(maxPAmgSpec[i] > best_p_amg_sp){
            best_p_amg_sp = maxPAmgSpec[i];
            possSpec = static_cast<ESpecies>(i);
        }
    }

    if( best_p_amg_sp < 0.5){
        possSpec = SPECIES_BLACK_STORK;
    }
    return possSpec;
}

} /*namespace ducks*/