#ifndef _CHECKERS_ZOBRIST_HPP_
#define _CHECKERS_ZOBRIST_HPP_
#define ZOBRIST_SIZE 34   // 0..31 board; 32 next player; 33 for moves_left
#define TYPES 8
#define MAXHASHSIZE 100000    // this is used to limit the size of hash map
#include <iostream>
#include <random>
#include <array>
#include "gamestate.hpp"
#include "constants.hpp"
// reference:
// https://www.geeksforgeeks.org/minimax-algorithm-in-game-theory-set-5-zobrist-hashing/

namespace checkers{

    // ZobristHashing is a singleton.
    // ZobristHashing initializes once and only once
    class ZobristHashing
    {
        public:
            static ZobristHashing& getInstance() {
                static ZobristHashing instance;
                return instance;
            }

        ZobristHashing(const ZobristHashing&) = delete;
        ZobristHashing(ZobristHashing&&) = delete;
        ZobristHashing& operator=(const ZobristHashing&) = delete;
        ZobristHashing& operator=(ZobristHashing&&) = delete;


        size_t hash(const GameState& state){

            // hash the game board
            size_t ret = hashBoard(state);

            // hash the next player
            ret ^= table[32][(int)state.getNextPlayer()];

            // hash the moves_left
            ret ^= table[33][(int)state.getMovesUntilDraw()];

            return ret;
        };

        size_t hashBoard(const GameState& state){
            size_t ret = 0;
            // hash the game board
            for (int i = 0; i < state.cSquares; ++i){
                uint8_t c = state.at(i);
                if (c == CELL_EMPTY) continue;
                if (c == CELL_INVALID) continue;
                ret ^= table[i][(int)c];
            }
            return ret;

        }

        private:
            // constructor
            ZobristHashing():mt(123){

                // assign the random value to the table
                for (int i = 0; i < ZOBRIST_SIZE; ++i){
                    for (int j = 0; j < TYPES; ++j){
                        table[i][j] = randomUInt64();
                    }
                }
            };
            ~ZobristHashing(){};


            // table
            std::array<std::array<size_t, TYPES>, ZOBRIST_SIZE> table;

            // random devices
            std::mt19937 mt;

            // function to generate random numbers
            size_t randomUInt64(){
                std::uniform_int_distribution<size_t> dist(0, UINT64_MAX);
                return dist(mt);
            }
    };


    struct GameStateHasher {
        std::size_t operator()(const GameState& g) const {
            ZobristHashing& z = ZobristHashing::getInstance();
            return z.hash(g) % MAXHASHSIZE;
        }
    };

    struct GameStateEqual {
        bool operator()(const GameState& lhs, const GameState& rhs) const
        {
            // compare board
            for (int i = 0; i < lhs.cSquares; ++i){
                if (lhs.at(i) != rhs.at(i)) return false;
            }

            // compare next player
            if (lhs.getNextPlayer() != rhs.getNextPlayer()) return false;

            // compare moves_left
            if (lhs.getMovesUntilDraw() != rhs.getMovesUntilDraw()) return false;

            // now lhs and rhs are equal.
            return true;
        }
    };

    struct GameBoardHasher {
        std::size_t operator()(const GameState& g) const {
            ZobristHashing& z = ZobristHashing::getInstance();
            return z.hashBoard(g) % MAXHASHSIZE;
        }
    };

    struct GameBoardEqual {
        bool operator()(const GameState& lhs, const GameState& rhs) const
        {
            // compare board
            for (int i = 0; i < lhs.cSquares; ++i){
                if (lhs.at(i) != rhs.at(i)) return false;
            }
            return true;
        }
    };
} // end checker namespace
#endif