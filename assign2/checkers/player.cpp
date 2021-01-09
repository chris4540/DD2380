#include "player.hpp"
#include <cstdlib>
#include <algorithm>
// reference:
// pseudocode:
//      http://people.csail.mit.edu/plaat/mtdf.html#abmem
// Japanese implementation in python (with explanation)
//      http://www.geocities.jp/m_hiroi/light/pyalgo26.html
namespace checkers
{

Player::Player(){
    maxDepth = 8;
    timeLimit = 0.05;
};

GameState Player::play(const GameState &pState,const Deadline &pDue)
{

    std::vector<GameState> lNextStates;
    pState.findPossibleMoves(lNextStates);

    if (lNextStates.size() == 0) return GameState(pState, Move());
    if (lNextStates.size() == 1) return lNextStates[0];

    // check if game state is in our map
    // GameState nextState;
    // if (getGameStateDecision(pState, nextState)){
    //     return nextState;
    // }

    // copy the deadline object for our instance method:
    // isTimesUp
    due_ = pDue;

    // check the players' color
    myColor = pState.getNextPlayer();
    opColor = myColor ^ (CELL_RED | CELL_WHITE);


    // decide the depth we would go
    int depth = maxDepth;
    if (pState.getMovesUntilDraw() <= maxDepth + 4) {
        // if near the end, go to the end
        depth = pState.getMovesUntilDraw();
    }

#if 1
    int maxIdx = -1;
    int nNextStates = lNextStates.size();
    int guess = 0;
    // perform iterative deepening searching with alpha beta, 21 score
    for (int d = 1; d < (depth+1); ++d){
        if (isTimesUp()) break;
        int maxGuess = INT_MIN;  // reset the maxGuess at every depth
        for (int i = 0; i < nNextStates; ++i) {
            if (isTimesUp()) break;
            guess = alphaBeta(lNextStates[i], d, INT_MIN, INT_MAX);
            if (guess >= maxGuess) {
                maxIdx = i;
                maxGuess = guess;
            }
        }
        if (isTimesUp()) break;
    }

#else
    int maxIdx = -1;
    int nNextStates = lNextStates.size();
    int guess;
    int maxGuess = INT_MIN;  // reset the maxGuess at every depth
    // perform iterative deepening searching with MTDF, 20 score
    for (int i = 0; i < nNextStates; ++i) {
        guess = 0;
        for (int d = 1; d < (depth+1); ++d){
            guess = MTDF(lNextStates[i], guess, d);
        }
        if (guess >= maxGuess) {
            maxIdx = i;
            maxGuess = guess;
        }
    }
#endif

    // save the decision
    // storeGameStateDecision(pState, lNextStates[maxIdx]);
    // storeGameStateDecision(pState.reversed(), lNextStates[maxIdx].reversed());
    return lNextStates[maxIdx];
}

int Player::MTDF(const GameState& pState, int fguess, int depth){
    // https://en.wikipedia.org/wiki/MTD-f
    int guess = fguess;
    int lower = INT_MIN;
    int upper = INT_MAX;
    do {
        int beta = max(guess, lower + 1);
        guess = alphaBeta(pState, depth, beta - 1, beta);
        if (guess < beta) {
            upper = guess;
        } else {
            lower = guess;
        }
    } while (lower < upper);
    return guess;
}

bool Player::isTimesUp() const {
    return (due_.now() > (due_ - timeLimit));
}

int Player::alphaBeta(const GameState& pState, int depth, int alpha, int beta){

    if (depth == 0 || pState.isEOG()){
        return getHeuristic(pState);
    }

    // search in the transposition table
    decltype(transPosTable)::const_iterator it = transPosTable.find(pState);
    if (it != transPosTable.end()) {
        // found in transposition table
        transPosEntry entry = it->second;
        if (entry.depth >= depth) {
            // if the entry found in transposition table is inside
            // the windows [alpha, beta] use it
            // entry.upper == entry.lower is the exact estimation of this node
            if (entry.lower >= beta) return entry.lower;
            if (entry.upper <= alpha || entry.upper == entry.lower) {
                return entry.upper;
            }
            // if not, update alpha and beta from the transposition table
            alpha = max(alpha, entry.lower);
            beta = min(beta, entry.upper);
        }
    }

    // declare the reture value
    int v;

    // get the child states
    vector<GameState> lNextStates;
    pState.findPossibleMoves(lNextStates);

    // if the player is me; max player
    if (pState.getNextPlayer() == this->myColor){
        v = INT_MIN;
        int a = alpha; // save original alpha value

        // sorting by the heuristic value from max to min
        sort(lNextStates.begin(), lNextStates.end(),
            [this](const GameState& a, const GameState& b) -> bool {
                return getHeuristic(a) > getHeuristic(b);}
            );

        // iterator over the vector
        for (GameState s: lNextStates) {
            v = max(v, alphaBeta(s, depth - 1, a, beta));
            a = max(a, v);
            if (beta <= a) {
                // beta pruning
                break;
            }
        }
    } else {
        // if the player is my opponent; min player
        v = INT_MAX;
        int b = beta;  // save original beta value

        // sorting by the heuristic value from min to max
        sort(lNextStates.begin(), lNextStates.end(),
            [this](const GameState& a, const GameState& b) -> bool {
                return getHeuristic(a) < getHeuristic(b);}
            );


        // iterator over the vector
        for (GameState s: lNextStates) {
            v = min(v, alphaBeta(s, depth - 1, alpha, b));
            b = min(b, v);
            if (b <= alpha) {
                // alpha pruning
                break;
            }
        }
    }

    // update the transposition table
    // we store the current state estimation and the reversed state estimation
    transPosEntry entry;
    transPosEntry revStateEntry;

    entry.depth = depth;
    revStateEntry.depth = depth;

    GameState revState = pState.reversed();

    if (v <= alpha) {
        entry.upper = v;
        entry.lower = INT_MIN;
        transPosTable[pState] = entry;

        revStateEntry.upper = INT_MAX;
        revStateEntry.lower = v;
        transPosTable[revState] = revStateEntry;
    } else if (v >= beta) {
        entry.upper = INT_MAX;
        entry.lower = v;
        transPosTable[pState] = entry;

        revStateEntry.upper = v;
        revStateEntry.lower = INT_MIN;
        transPosTable[revState] = revStateEntry;
    } else {
        entry.upper = v;
        entry.lower = v;
        transPosTable[pState] = entry;
        transPosTable[revState] = entry;
    }
    return v;
}

int Player::getHeuristic(const GameState& pState) {
    // this function calculates the heuristic in the view of white.
    // will consider the side when returning value

    int ret = 0;

    // check if End of game
    if (pState.isEOG()) {
        if (pState.isDraw()) { return 0; } // zero-sum game
        if (pState.isWhiteWin()) {
            ret = 1e7;
        } else {
            ret = -1e7;
        }
    } else {
        // find the game board in hash map
        decltype(boardToHVal)::const_iterator it = boardToHVal.find(pState);
        if (it != boardToHVal.end()) {
            ret = it->second;
        } else {
            // cannot find the board in boardToHVal, calculate the heuristic as I am white
            ret = 0;
            int kingScore = 1;
            int chkScore = 1;
            for (int i = 0; i < pState.cSquares; ++i){
                uint8_t c = pState.at(i);
                if (c == CELL_EMPTY) continue;
                if (c == CELL_INVALID) continue;
                if (c == (CELL_WHITE | CELL_KING)) ret += kingScore;
                if (c == CELL_WHITE) ret += chkScore;
                if (c == (CELL_RED | CELL_KING)) ret -= kingScore;
                if (c == CELL_RED) ret -= chkScore;
            }

            // save the value
            boardToHVal[pState] = ret;
            // boardToHVal[pState.reversed()] = -ret;
        }
    }

    if (myColor == CELL_WHITE) {
        return ret;
    } else {
        return -ret;
    }
}

bool Player::getGameStateDecision(const GameState& curState, GameState& nextState) const{
    string nextStateMsg;

    decltype(gsToNext)::const_iterator it = gsToNext.find(curState);
    if (it != gsToNext.end()) {
        nextStateMsg = it->second;
        nextState = GameState(nextStateMsg);
        return true;
    }
    return false;
}

void Player::storeGameStateDecision(const GameState& curState, const GameState& nextState){
    gsToNext[curState] = nextState.toMessage();
}

/*namespace checkers*/ }
