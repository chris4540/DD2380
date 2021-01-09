#include "player.hpp"
#include <cstdlib>
#include <algorithm>
#include <unistd.h>
#include <climits>

using namespace std;
namespace TICTACTOE
{

Player::Player(){
    NumCells = 16;
    NumRows = 4;
    NumCols = 4;
    MaxDepth = 16;
    winVector = getWinVectors();
    myCell = CELL_INVALID;
    opCell = CELL_INVALID;
}

GameState Player::play(const GameState &pState,const Deadline &pDue)
{

    // Initialize myCell; first action must belong to me!
    if (myCell == CELL_INVALID){
        myCell = pState.getNextPlayer();
        opCell = myCell ^ (CELL_X | CELL_O);
    }

    vector<GameState> lNextStates;
    pState.findPossibleMoves(lNextStates);

    if (lNextStates.size() == 0) return GameState(pState, Move());
    if (lNextStates.size() == 1) return lNextStates[0];
    /*
     * Here you should write your clever algorithms to get the best next move, ie the best
     * next state. This skeleton returns a random move instead.
     */
    // return lNextStates[rand() % lNextStates.size()];

    int maxHeuristicValue = INT_MIN;
    int maxIdx = -1;

    for (int i = 0; i < (int)lNextStates.size(); ++i){

        // calculate depth
        int depth;
        if (getPossibleMoves(pState) > MaxDepth) {
            depth = MaxDepth;
        } else {
            depth = MaxDepth + 1;
        }

        int value = alphaBeta(lNextStates[i], depth, INT_MIN, INT_MAX);
        if (value > maxHeuristicValue) {
            maxHeuristicValue = value;
            maxIdx = i;
        }
    }

    return lNextStates[maxIdx];
}

int Player::alphaBeta(const GameState& pState, int depth, int alpha, int beta){
    // recurivly call this function to get the bestPossible
    // return a heuristic value that approximates a utility function of the state

    if (depth == 0 || pState.isEOG()){
        return getHeuristic(pState);
    }

    int v;
    // get the child states
    vector<GameState> lNextStates;
    pState.findPossibleMoves(lNextStates);

    // if the player is me
    if (pState.getNextPlayer() == this->myCell){
        v = INT_MIN;

        // iterator over the vector
        for (int i = 0; i < (int)lNextStates.size(); ++i){
            GameState s = lNextStates[i];
            v = max(v, alphaBeta(s, depth - 1, alpha, beta));
            alpha = max(alpha, v);
            if (beta <= alpha) {
                // beta pruning
                break;
            }
        }
    } else {
        // if the player is my opponent
        v = INT_MAX;

        // iterator over the vector
        for (int i = 0; i < (int)lNextStates.size(); ++i){
            GameState s = lNextStates[i];
            v = min(v, alphaBeta(s, depth - 1, alpha, beta));
            beta = min(beta, v);
            if (beta <= alpha) {
                // alpha pruning
                break;
            }
        }
    }
    return v;
}

int Player::getHeuristic(const GameState& pState){

    int ret = 0;
    for (uint i = 0; i < winVector.size(); ++i){
        int myWinPoss = 0;  // the number of possibilities of me
        int opWinPoss = 0;  // the number of possibilities of my opponent
        for (int j = 0; j < NumCols; ++j){
            uint8_t cell = pState.at(winVector[i][j]);
            if (cell == myCell){
                myWinPoss++;
            }
            if (cell == opCell) {
                opWinPoss++;
            }
        }
        ret += Heuristic_Array[myWinPoss][opWinPoss];
    }
    return ret;
}

int Player::getPossibleMoves(const GameState& pState) {
    int ret = 0;
    for (int i = 0; i < NumCells; i++){
        if (pState.at(i) == CELL_EMPTY) {
            ret++;
        }
    }
    return ret;
}


// This function returns all the possible winning vectors
// author: george
vector< vector<int> > Player::getWinVectors() {
    string Board;
    for (int i(0); i<NumRows; i++) {
        for (int j(0); j<NumCols; j++) {
            Board += "0";
        }
    }
    char pl;
    pl = '0';

    vector< vector<int> > w;

    // winning Row
    int SumR;
    for (int i(0); i<NumRows; i++) {
        SumR = 0;
        vector<int> temp_w;
        for (int j(0); j<NumCols; j++) {
            if (Board[i*NumCols+j] == pl) {
                SumR++;
                temp_w.push_back(i*NumCols+j);
            }
        }
        if (SumR == 4) {
            w.push_back(temp_w);
        }
    }
    // winning Col
    int SumC;
    for (int j(0); j<NumCols; j++) {
        SumC = 0;
        vector<int> temp_w;
        for (int i(0); i<NumRows; i++) {
            if (Board[i*NumCols+j] == pl) {
                SumC++;
                temp_w.push_back(i*NumCols+j);
            }
        }
        if (SumC == 4) {
            w.push_back(temp_w);
        }
    }
    // wining Diagonal & 2nd Diagonal
    int SumD, SumD_;
    SumD = 0;
    SumD_ = 0;
    vector<int> temp_w;
    vector<int> temp_w_;
    for (int i(0); i<NumRows; i++) {
        for (int j(0); j<NumCols; j++) {
            if (Board[i*NumCols+j] == pl && i == j) {
                SumD++;
                temp_w.push_back(i*NumCols+j);
            }
            if (Board[i*NumCols+j] == pl && i+j == NumRows-1) {
                SumD_++;
                temp_w_.push_back(i*NumCols+j);
            }
        }
    }
    if (SumD == NumCols) {
        w.push_back(temp_w);
    }
    if (SumD_ == NumCols) {
        w.push_back(temp_w_);
    }

    return w;
}

/*namespace TICTACTOE*/ }
