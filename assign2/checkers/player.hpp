#ifndef _CHECKERS_PLAYER_HPP_
#define _CHECKERS_PLAYER_HPP_

#include "constants.hpp"
#include "deadline.hpp"
#include "move.hpp"
#include "gamestate.hpp"
#include "zobrist.hpp"
#include <vector>
#include <unordered_map>
#include <climits>

using namespace std;
namespace checkers
{

    // Transposition Table Entry
    struct transPosEntry {
        int depth = INT_MIN;
        int upper = INT_MAX;
        int lower = INT_MIN;
    };

class Player
{
public:
    Player();
    ///perform a move
    ///\param pState the current state of the board
    ///\param pDue time before which we must have returned
    ///\return the next state the board is in after our move
    GameState play(const GameState &pState, const Deadline &pDue);
private:
    int maxDepth;
    uint8_t myColor;
    uint8_t opColor;
    Deadline due_;
    double timeLimit;

    // game tree searching routines
    int alphaBeta(const GameState& pState, int depth, int alpha, int beta);

    // map board to heuristic value
    unordered_map<GameState, int, GameBoardHasher, GameBoardEqual> boardToHVal;
    int getHeuristic(const GameState& pState);

    // game state to next state
    unordered_map<GameState, string, GameStateHasher, GameStateEqual> gsToNext;
    bool getGameStateDecision(const GameState& curState, GameState& nextState) const;
    void storeGameStateDecision(const GameState& curState, const GameState& nextState);


    // transposition table for alpha-beta pruning
    unordered_map<GameState, transPosEntry, GameStateHasher, GameStateEqual> transPosTable;

    int MTDF(const GameState& pState, int fguess, int depth);
    // int iterative_deepening(const GameState&, int);
    bool isTimesUp() const;

};

/*namespace checkers*/ }

#endif
