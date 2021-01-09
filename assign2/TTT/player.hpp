#ifndef _TICTACTOE_PLAYER_HPP_
#define _TICTACTOE_PLAYER_HPP_

#include "constants.hpp"
#include "deadline.hpp"
#include "move.hpp"
#include "gamestate.hpp"
#include <vector>

namespace TICTACTOE
{

class Player
{
public:
    ///perform a move
    ///\param pState the current state of the board
    ///\param pDue time before which we must have returned
    ///\return the next state the board is in after our move
    GameState play(const GameState &pState, const Deadline &pDue);

    Player();
private:
    int NumCells;
    int NumRows;
    int NumCols;
    int MaxDepth;
    uint8_t myCell;
    uint8_t opCell;
    int getPossibleMoves(const GameState& pState);
    int alphaBeta(const GameState& pState, int depth, int alpha, int beta);
    int getHeuristic(const GameState& pState);
    std::vector< std::vector<int>> getWinVectors();
    std::vector< std::vector<int>> winVector;

    const int Heuristic_Array[5][5] = {
        {     0,   -10,  -100, -1000, -10000},
        {    10,     0,     0,     0,      0},
        {   100,     0,     0,     0,      0},
        {  1000,     0,     0,     0,      0},
        { 10000,     0,     0,     0,      0}
    };
};

/*namespace TICTACTOE*/ }

#endif
