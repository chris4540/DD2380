#include "HmmDHModels.hpp"
#include "Constants.hpp"

using namespace hmm;
int main(){
    HmmDHModels* models = HmmDHModels::getInstance();


    HMM* hmm = (*models).getModelBySpecies(ducks::SPECIES_PIGEON);

    (*hmm).printOutModelParam();
}