#include <iostream>
#include <unordered_map>
#include "zobrist.hpp"


using namespace checkers;
using namespace std;

int main()
{
    ZobristHashing& z = ZobristHashing::getInstance();

    uint64_t hashValue;
    uint64_t hashBoard;
    GameState g1("rrrrrr.rrr....rr....www.wwwwwwww 1_6_15 w 50");
    hashValue = z.hash(g1);
    hashBoard = z.hashBoard(g1);
    cout << hashValue << endl;
    cout << hashBoard << endl;



    GameState g2("rrrrrr.rrr....rr....www.wwwwwwww 1_6_15 r 50");
    hashValue = z.hash(g2);
    hashBoard = z.hashBoard(g2);
    cout << hashValue << endl;
    cout << hashBoard << endl;


    GameState g3("rrrrrr.rrr....rr....www.wwwwwwww 1_6_15 w 40");
    hashValue = z.hash(g3);
    hashBoard = z.hashBoard(g3);
    cout << hashValue << endl;
    cout << hashBoard << endl;

    typedef unordered_map<GameState, int, GameStateHasher, GameStateEqual> GameHashMap;
    GameHashMap mapping{ {g1, 1}, {g2, 2}, {g3, 3} };

    for (auto &kv : mapping) {
        cout << kv.first.toMessage() << ":";
        cout << kv.second << endl;
    }

    // same as g3, but not the last move
    GameState g4("rrrrrr.rrr....rr....www.wwwwwwww 0_31_27 w 40");
    decltype(mapping)::const_iterator it = mapping.find(g4);
    if (it != mapping.end()) {
        cout << "Found g4!" << endl;
        cout << it->second << endl;
    }

    return 0;
}