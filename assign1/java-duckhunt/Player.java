import java.util.ArrayList;


class Player {

    final int emissions = 9;
    final int states = 5;
    final int species = Constants.COUNT_SPECIES;

    ArrayList<ArrayList> models =  new ArrayList<ArrayList>();

    int numberBirds;
    int currentRound = 0;
    int start =2;
    int time = 0;

    public Player() {
        
        //initialize our models
        for (int i = 0; i < species; i++) {
            models.add(i, new ArrayList<HMM>());
        }
    }

    /**
     * Shoot!
     *
     * This is the function where you start your work.
     *
     * You will receive a variable pState, which contains information about all
     * birds, both dead and alive. Each bird contains all past moves.
     *
     * The state also contains the scores for all players and the number of
     * time steps elapsed since the last time this function was called.
     *
     * @param pState the GameState object with observations etc
     * @param pDue time before which we must have returned
     * @return the prediction of a bird we want to shoot at, or cDontShoot to pass
     */
    public Action shoot(GameState pState, Deadline pDue) {


        this.time++;
        if(pState.getRound() > this.currentRound){
            this.currentRound = pState.getRound();
            this.time = 0;
        }


        if( time < 75){
            return cDontShoot;
        }

        if( currentRound > 0) {
            return getNextMove(pState);
        }
        return cDontShoot;
    }

    public Action getNextMove(GameState pState){


        double bestProbOverall = 0;
        int bestMoveOverall = -1;
        int bestBird = -1;

        for( int i = 0; i < pState.getNumBirds(); i++) {

            if (pState.getBird(i).isDead()) {
                continue;
            }

            int species = getLikelySpecies(pState.getBird(i));
            if (species == Constants.SPECIES_BLACK_STORK) {
                continue;
            }



//            double[] probArray = new double[Constants.COUNT_MOVE];

            int[] obs = getObservations(pState.getBird(i));

            HMM moveJ = new HMM(states, emissions);
            moveJ.estimateModel(obs);

            double[] stateDistribution = moveJ.estimateStateDistribution(obs);
            stateDistribution = normalize(stateDistribution);
            double[] currentEmissions = moveJ.estimateProbabilityDistributionOfNextEmission(stateDistribution);
            currentEmissions = normalize(currentEmissions);

            double bestProb = 0;
            int bestMove = 0;

            for (int j = 0; j < currentEmissions.length; j++) {
                if (currentEmissions[j] > bestProb) {
                    bestProb = currentEmissions[j];
                    bestMove = j;
                }
            }



           /* int[] obsNext = new int[obs.length+1];
            for(int k = 0; k < obs.length; k++){
                obsNext[k] = obs[k];
            }
            obsNext[obsNext.length - 1] = bestMove;

            double[] nextStateDistribution = moveJ.estimateStateDistribution(obsNext);
            nextStateDistribution = normalize(nextStateDistribution);
            double[] nextEmissions = moveJ.estimateProbabilityDistributionOfNextEmission(nextStateDistribution);
            nextEmissions = normalize(nextEmissions);

            double bestProbNext = 0;
            int bestMoveNext = 0;

            for (int j = 0; j < nextEmissions.length; j++) {
                if (nextEmissions[j] > bestProbNext) {
                    bestProbNext = nextEmissions[j];
                    bestMoveNext = j;
                }
            }
            



            //this is for all birds
            if( bestProbNext > bestProbOverall){
                bestProbOverall = bestProbNext;
                bestMoveOverall = bestMoveNext;
                bestBird = i;
            }*/

            //this is for all birds
            if( bestProb > bestProbOverall){
                bestProbOverall = bestProb;
                bestMoveOverall = bestMove;
                bestBird = i;
            }
        }


        return new Action(bestBird, bestMoveOverall);

//        return cDontShoot;

    }

    public int getLikelySpecies(Bird bird) {
        //for each one of our models(for each species), return the one that maximises the probability of the observation sequence(for our bird) given the model.

        if(this.currentRound == 0){
            return 0;
//          return  (int)(Math.random() * 5);

        }

        double bestOverallP = 0;
        int bestSpecies = 0;
        double[] max  = new double[species];
        double[] average = new double[species];

        for (int i = 0; i < species; i++) {

            double bestP = 0;
            ArrayList<HMM> speciesArray = this.models.get(i);


            double avg = 0;

            for(int j = 0; j < speciesArray.size(); j++){

                double p = speciesArray.get(j).estimateProbabilityOfEmissionSequence(getObservations(bird));
                avg += p;

//                System.err.println("P :"+ p);
                if (bestP < p) {
                    bestP = p;
                }
            }

            average[i] = (double) avg / speciesArray.size();
            max[i] = bestP;

        }

        max  = normalize(max);

        for(int i  = 0; i < max.length; i++){
            if( max[i] > bestOverallP ){
                bestOverallP = max[i];
                bestSpecies = i;
            }
        }


            if( bestOverallP < 0.5){
                bestSpecies = 5;
           }
       


        return bestSpecies;
    }

    public int[] getObservations(Bird bird) {
        ArrayList<Integer> obsList = new ArrayList<Integer>();
        for (int i = 0; i < bird.getSeqLength(); i++) {
            if (bird.wasDead(i)) {
                break;
            }
            else {
                obsList.add(bird.getObservation(i));
            }
        }

        int[] obs = new int[obsList.size()];
        for (int i = 0; i < obsList.size(); i++) {
            obs[i] = obsList.get(i);
        }

        return obs; 
    }

    public void printObservations(int[] observations) {
        System.err.println("OBSERVATIONS");
        for (int i = 0; i < observations.length; i++) {
            System.err.print(observations[i]);
        }
        System.err.println("Size: "+ observations.length );

    }


    /**
     * Guess the species!
     * This function will be called at the end of each round, to give you
     * a chance to identify the species of the birds for extra points.
     *
     * Fill the vector with guesses for the all birds.
     * Use SPECIES_UNKNOWN to avoid guessing.
     *
     * @param pState the GameState object with observations etc
     * @param pDue time before which we must have returned
     * @return a vector with guesses for all the birds
     */
    public int[] guess(GameState pState, Deadline pDue) {
        /*
         * Here you should write your clever algorithms to guess the species of
         * each bird. This skeleton makes no guesses, better safe than sorry!
         */

        int[] lGuess = new int[pState.getNumBirds()];
        for (int i = 0; i < pState.getNumBirds(); ++i) {
            Bird bird = pState.getBird(i);
            lGuess[i] = getLikelySpecies(bird);
           // lGuess[i] = Constants.SPECIES_UNKNOWN;
        }            
        return lGuess;


    }

    /**
     * If you hit the bird you were trying to shoot, you will be notified
     * through this function.
     *
     * @param pState the GameState object with observations etc
     * @param pBird the bird you hit
     * @param pDue time before which we must have returned
     */
    public void hit(GameState pState, int pBird, Deadline pDue) {
        System.err.println("HIT BIRD!!!");
    }

    /**
     * If you made any guesses, you will find out the true species of those
     * birds through this function.
     *
     * @param pState the GameState object with observations etc
     * @param pSpecies the vector with species
     * @param pDue time before which we must have returned
     */
    public void reveal(GameState pState, int[] pSpecies, Deadline pDue) {

        for (int i = 0; i < pSpecies.length; i++) {

            Bird bird = pState.getBird(i);
//            System.err.println("LENGTH BIRD"+i+ "  "+getObservations(bird).length);
            int species = pSpecies[i];

            for(int j = 0; j < 8; j++){
                HMM birdModel = new HMM(states, emissions);
//            birdModel.printMatrix();
                birdModel.estimateModel(getObservations(bird));

                this.models.get(species).add(birdModel);

            }

        }


    }

    public static final Action cDontShoot = new Action(-1, -1);

    public static double[] normalize(double[] a) {

        double sum =0.0;
        double[] a2 = new double[a.length];
        for (int i = 0; i < a.length; i++) {
          sum += a[i];
        }
        //System.err.println("SUM "+ sum);
        for (int i = 0; i < a.length; i++) {
          a2[i] = a[i] * (1.0 / sum);
        }

        return a2;
    }




}
