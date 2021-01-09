#ifndef _TICTACTOE3D_PLAYER_HPP_
#define _TICTACTOE3D_PLAYER_HPP_

#include "constants.hpp"
#include "deadline.hpp"
#include "move.hpp"
#include "gamestate.hpp"
#include <vector>
#include <map>

using namespace std;
namespace TICTACTOE3D
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
    int numInARow;
    int MaxDepth;
    uint8_t myCell;
    uint8_t opCell;
    double TimeLimit;

    // A hash map which map the string game state to a heurisitc value
    // this hash map will be used in "getHeuristic"
    map<string, int> mapGameConfigToHVal;

    // Get the game configuration string
    // @staticmethod
    // @param pState: the game state instance
    // @return the string representing the game configuration
    // @return.example:
    //      x..ox..ox...x..o.....ox..ox..........xo.ox......o..x..........ox
    string getGameConfig(const GameState& pState) const;

    int getHeuristic(const GameState& pState);
    int alphaBeta(const GameState& pState, int depth, int alpha, int beta);
    int getPossibleMoves(const GameState& pState);
    vector< vector<int> > winVector = {
        {0, 1, 2, 3},
        {0, 4, 8, 12},
        {0, 16, 32, 48},
        {4, 5, 6, 7},
        {1, 5, 9, 13},
        {16, 17, 18, 19},
        {16, 20, 24, 28},
        {1, 17, 33, 49},
        {4, 20, 36, 52},
        {8, 9, 10, 11},
        {2, 6, 10, 14},
        {32, 33, 34, 35},
        {32, 36, 40, 44},
        {2, 18, 34, 50},
        {8, 24, 40, 56},
        {12, 13, 14, 15},
        {3, 7, 11, 15},
        {48, 49, 50, 51},
        {48, 52, 56, 60},
        {3, 19, 35, 51},
        {12, 28, 44, 60},
        {20, 21, 22, 23},
        {17, 21, 25, 29},
        {5, 21, 37, 53},
        {24, 25, 26, 27},
        {18, 22, 26, 30},
        {36, 37, 38, 39},
        {33, 37, 41, 45},
        {6, 22, 38, 54},
        {9, 25, 41, 57},
        {28, 29, 30, 31},
        {19, 23, 27, 31},
        {52, 53, 54, 55},
        {49, 53, 57, 61},
        {7, 23, 39, 55},
        {13, 29, 45, 61},
        {40, 41, 42, 43},
        {34, 38, 42, 46},
        {10, 26, 42, 58},
        {44, 45, 46, 47},
        {35, 39, 43, 47},
        {56, 57, 58, 59},
        {50, 54, 58, 62},
        {11, 27, 43, 59},
        {14, 30, 46, 62},
        {60, 61, 62, 63},
        {51, 55, 59, 63},
        {15, 31, 47, 63},
        {0, 21, 42, 63},
        {12, 25, 38, 51},
        {3, 22, 41, 60},
        {15, 26, 37, 48},
        {0, 5, 10, 15},
        {3, 6, 9, 12},
        {16, 21, 26, 31},
        {19, 22, 25, 28},
        {32, 37, 42, 47},
        {35, 38, 41, 44},
        {48, 53, 58, 63},
        {51, 54, 57, 60},
        {0, 20, 40, 60},
        {1, 21, 41, 61},
        {2, 22, 42, 62},
        {3, 23, 43, 63},
        {12, 24, 36, 48},
        {13, 25, 37, 49},
        {14, 26, 38, 50},
        {15, 27, 39, 51},
        {0, 17, 34, 51},
        {4, 21, 38, 55},
        {8, 25, 42, 59},
        {12, 29, 46, 63},
        {3, 18, 33, 48},
        {7, 22, 37, 52},
        {11, 26, 41, 56},
        {15, 30, 45, 60}};

    const int Heuristic_Array[5][5] = {
        {     0,   -10,  -100, -1000, -10000},
        {    10,     0,     0,     0,      0},
        {   100,     0,     0,     0,      0},
        {  1000,     0,     0,     0,      0},
        { 10000,     0,     0,     0,      0}
    };

};

/*namespace TICTACTOE3D*/ }

#endif
