#include "player.hpp"
#include <cstdlib>
#include <algorithm>
#include <unistd.h>
#include <climits>

using namespace std;

namespace TICTACTOE3D
{

Player::Player(){
    NumCells = 64;
    numInARow = 4;
    MaxDepth = 2;
    myCell = CELL_INVALID;
    opCell = CELL_INVALID;
    TimeLimit = 0.2;
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

        if (pDue.now() > (pDue - TimeLimit)) {
            break;
        }
        // -----------------------------------------------
        // TODO: implement the dynamic depth according to the empty cells
        // calculate depth
        int depth;
        if (getPossibleMoves(pState) > MaxDepth) {
            depth = MaxDepth;
        } else {
            depth = MaxDepth + 1;
        }
        // -----------------------------------------------

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

    // check if the state is in the hash map which maps game state to
    // heuristic value
    string config = getGameConfig(pState);

    // get the iterator by trying to find it among keys
    map<string, int>::const_iterator pos = mapGameConfigToHVal.find(config);
    if (pos != mapGameConfigToHVal.end()){  // if we found it!
        // find the configuration in hash map!
        ret =  pos->second;   // set the ret to the heuristic value of the game state
        return ret;
    }


    // calculate the heuristic value
    for (uint i = 0; i < winVector.size(); ++i){
        int myMarkInWinVec = 0;  // the number of my marks in this win vector
        int opMarkInWinVec = 0;  // the number of opponent marks in this win vector
        for (int j = 0; j < numInARow; ++j){
            uint8_t cell = pState.at(winVector[i][j]);
            if (cell == myCell){
                myMarkInWinVec++;
            }
            if (cell == opCell) {
                opMarkInWinVec++;
            }
        }
        ret += Heuristic_Array[myMarkInWinVec][opMarkInWinVec];
    }

    // save down the heuristic value
    mapGameConfigToHVal[config] = ret;
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

string Player::getGameConfig(const GameState& pState) const{
    string ret;
    stringstream ss(pState.toMessage());
    // stop at the first space
    ss >> ret;
    return ret;
}
/*namespace TICTACTOE3D*/ }
