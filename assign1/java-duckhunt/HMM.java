import java.lang.Math;
/**
 * Represents a HMM.
 *
 * Much of the terminology is borrowed from a very pedagogic paper by Rabiner 
 * and Juang: http://www.cs.ubc.ca/~murphyk/Bayes/rabiner.pdf
 *
 * We recommend that you read the article as an introduction to HMMs.
 */
public class HMM {

  static final int maxIters = 30; // Max iterations when estimating a new model.
  final int numberOfStates; // The number of states in the HMM.
  final int numberOfEmissions; // The number of emissions in the HMM.
  
  double[][] A; // The transition matrix of the HMM.
  double[][] B; // The emission matrix of the HMM.
  double[] pi; // The initial state distribution of the HMM.

  public HMM(int numberOfStates, int numberOfEmissions) {
    /**
     * This constructor just sets everything to zero (why is this wrong?).
     * Write a clever way to initialize the HMM!
     */

    this.numberOfStates = numberOfStates;
    this.numberOfEmissions = numberOfEmissions;
    this.A = new double[numberOfStates][numberOfStates];
    this.B = new double[numberOfStates][numberOfEmissions];
    this.pi = new double[numberOfStates];

    double sum = 0;
    double eps;
    for (int i = 0; i < numberOfStates; ++i) {
      for (int j = 0; j < numberOfStates; ++j) {
        this.A[i][j] = 1000 + Math.random()*10000;
//        this.A[i][j] = Math.random()*(0.9-0.1)+0.1;
       // A[i][j] = 3 + Math.random()*(0.09-0.01)+0.01; 
        //  if (i == j)
         //   A[i][j] += 100 * numberOfStates;
      }
    }

    this.A= normalize(A);
   

    for (int i = 0; i < numberOfStates; ++i) {
      for (int j = 0; j < numberOfEmissions; ++j) {
        this.B[i][j] = 1000 + Math.random()*500;
//        this.B[i][j] = Math.random()*(0.9-0.1)+0.1;
       // B[i][j] = 3 + Math.random()*(0.09-0.01)+0.01; 
          //if (i == j)
           // B[i][j] += 100 * numberOfStates;
          //if(j >= numberOfStates)
           // B[i][j] += 10;
      }
    }

    this.B = normalize(B);

    for (int i = 0; i < numberOfStates; ++i) {
      this.pi[i] = 1000 + Math.random()*500;
//      this.pi[i] = Math.random()*(0.9-0.1)+0.1;
      //this.pi[i] = 1 + Math.random()*(0.09-0.01)+0.01;
    }

    this.pi = normalize(pi);
    // printMatrix();
  }

  public HMM(double[][] A, double[][] B, double[] pi){
    this.numberOfStates = A[0].length;
    this.numberOfEmissions = B[0].length;
    this.A = new double[numberOfStates][numberOfStates];
    this.B = new double[numberOfStates][numberOfEmissions];
    this.pi = new double[numberOfStates];

    for (int i = 0; i < numberOfStates; ++i) {
      for (int j = 0; j < numberOfStates; ++j) {
        this.A[i][j] = A[i][j];
      }
    }
    
    for (int i = 0; i < numberOfStates; ++i) {
      for (int j = 0; j < numberOfEmissions; ++j) {
        this.B[i][j] = B[i][j];
      }
    }

    for (int i = 0; i < numberOfStates; ++i) {
      this.pi[i] = pi[i];
    }
  }
  
  /**
   * Estimates the probability distribution of the next emission, given the 
   * current state probability distribution.
   * 
   * Note that this method solves the preparatory exercise HMM1.
   * 
   * @param currentStateProbabilityDistribution the current state probability 
   *        distribution
   */
  public double[] estimateProbabilityDistributionOfNextEmission(
      double[] currentStateProbabilityDistribution){
    double[] probabilityOfMoves = new double[numberOfEmissions];
    for(int i = 0; i < numberOfStates; i++){
      for(int j = 0; j < numberOfStates; j++){
        for(int k = 0; k < numberOfEmissions; k++){
          probabilityOfMoves[k] += 
              currentStateProbabilityDistribution[j]*A[j][i]*B[i][k];
        }
      }
    }
    return probabilityOfMoves;
  }

  /**
   * Estimates the probability of a sequence of observed emissions, assuming
   * this HMM.
   * 
   * Note that this method solves the preparatory exercise HMM2.
   * 
   * @param O the sequence of observed emissions
   */
  public double estimateProbabilityOfEmissionSequence(int[] O){
    double probability = 0.0;
    double[][] alpha = new double[O.length][numberOfStates];

    for (int i = 0; i < numberOfStates; ++i) {
      alpha[0][i]=pi[i]*B[i][O[0]];
    }
    
    for (int t = 1; t < O.length; ++t) {
      for (int i = 0; i < numberOfStates; ++i) {
        for (int j = 0; j < numberOfStates; ++j) {
          alpha[t][i] += alpha[(t-1)][j]*A[j][i]*B[i][O[t]];
        }
      }
    }
   
    for (int i = 0; i < numberOfStates; ++i) {
      probability += alpha[(O.length-1)][i];
    }
    return probability;
  }
  

   public double[] estimateStateDistribution(int[] O){
    double probability = 0.0;
    double[][] alpha = new double[O.length][numberOfStates];
    double[] stateDistribution = new double[numberOfStates];

    for (int i = 0; i < numberOfStates; ++i) {
      alpha[0][i]=pi[i]*B[i][O[0]];
    }

    for (int t = 1; t < O.length; ++t) {
      for (int i = 0; i < numberOfStates; ++i) {
        for (int j = 0; j < numberOfStates; ++j) {
          alpha[t][i] += alpha[(t-1)][j]*A[j][i]*B[i][O[t]];
        }
      }
    }
    stateDistribution = alpha[O.length-1];
    return stateDistribution;
  }
  /**
   * Estimates the hidden states from which a sequence of emissions were
   * observed.
   * 
   * Note that this method solves the preparatory exercise HMM3.
   * 
   * @param O the sequence of observed emissions
   */
  public int[] estimateStateSequence(int[] O){
    
    double[][] alpha = new double[O.length][numberOfStates];
    int[][] path = new int[O.length][numberOfStates];
    int[] finalPath = new int[O.length];
    double maxProbability;
    
    for (int i = 0; i < numberOfStates; ++i) {
      alpha[0][i]=pi[i]*B[i][O[0]];
    }
    
    for (int t = 1; t < O.length; ++t) {
      for (int i = 0; i < numberOfStates; ++i) {
        maxProbability = 0.0;
        for (int j = 0; j < numberOfStates; ++j) {
          if (maxProbability < alpha[t-1][j]*A[j][i]){
            maxProbability = alpha[t-1][j]*A[j][i];
            path[t][i] = j;
          }
        }
        alpha[t][i] = maxProbability*B[i][O[t]];
      }
    }

    maxProbability = 0.0;
    for (int i = 0; i < numberOfStates; ++i) {
      if (maxProbability < alpha[O.length-1][i]){
        maxProbability = alpha[O.length-1][i];
        finalPath[O.length-1] = i;
      }
    }

    for (int t = O.length-2; t >= 0; --t) {
      finalPath[t] = path[t+1][finalPath[t+1]];
    }

    return finalPath;
  }

  /**
   * Re-estimates this HMM from a sequence of observed emissions.
   * 
   * Note that this method solves the preparatory exercise HMM4.
   * 
   * @param O the sequence of observed emissions
   */
  public void estimateModel(int O[]){
    double[][][] xi = new double[O.length][numberOfStates][numberOfStates];
    double[][] alpha = new double[O.length][numberOfStates];
    double[][] beta = new double[O.length][numberOfStates];
    double[][] gamma = new double[O.length][numberOfStates];

    double[] C = new double[O.length]; // A scaling factor
    double numer; // A temporary variable for holding a numerator
    double denom; // A temporary variable for holding a denominator

    /* Iteration-related stuff */
    double oldLogProb = -Double.MAX_VALUE;
    int iters = 0;
    double logProb;
    boolean finished = false;

    while(!finished && iters < maxIters){
      /* Computation of alpha */

      C[0]=0.0;
      for (int i = 0; i < numberOfStates; ++i) {
        alpha[0][i]=pi[i]*B[i][O[0]];
        C[0] += alpha[0][i];
      }
      
      if (C[0] != 0){
        C[0] = 1.0/C[0];
      }
      for (int i = 0; i < numberOfStates; ++i) {
        alpha[0][i]=C[0]*alpha[0][i];
      }

      for (int t = 1; t < O.length; ++t) {
        C[t]=0.0;
        for (int i = 0; i < numberOfStates; ++i) {
          alpha[t][i] = 0.0;
          for (int j = 0; j < numberOfStates; ++j) {
            alpha[t][i] += alpha[(t-1)][j]*A[j][i];
          }
          alpha[t][i]=alpha[t][i]*B[i][O[t]];
          C[t] += alpha[t][i];
        }
        
        if (C[t] != 0){
          C[t] = 1.0/C[t];
        }
        for (int i = 0; i < numberOfStates; ++i) {
          alpha[t][i]=C[t]*alpha[t][i];
        } 
      }
      
      /* Computation of beta */
        
      for (int i = 0; i < numberOfStates; ++i) {
        beta[(O.length-1)][i] = C[O.length-1];
      }
 
      for (int t = O.length-2; t >= 0; --t){
        for (int i = 0; i < numberOfStates; ++i) {
          beta[t][i] = 0;
          for (int j = 0; j < numberOfStates; ++j) {
            beta[t][i] += A[i][j]*B[j][O[t+1]]*beta[t+1][j];
          }
          beta[t][i] = C[t]*beta[t][i];
        }
      }

      /* Computation of gamma and xi */
      
      for (int t = 0; t < O.length-1; ++t) {
        denom = 0;
        for (int i = 0; i < numberOfStates; ++i) {
          for (int j = 0; j < numberOfStates; ++j){
            /* Eq. 37 in Rabiner89 */
            denom += alpha[t][i]*A[i][j]*B[j][O[t+1]]*beta[(t+1)][j];
          }
        }
        for (int i = 0; i < numberOfStates; ++i) {
          gamma[t][i] = 0.0;
          for (int j = 0; j < numberOfStates; ++j){
            if (denom != 0){
              xi[t][i][j] = (alpha[t][i]*A[i][j]*B[j][O[t+1]]*beta[(t+1)][j])/denom;
            } else {
              xi[t][i][j] = 0.0;
            }
            /* Eq. 38 in Rabiner89 */
            gamma[t][i] += xi[t][i][j];
          }
        }
      }
      /* We must also calculate gamma for the last step. This is given by Eq. 27
       * in Rabiner89. */
      denom = 0;
      for (int i = 0; i < numberOfStates; ++i) {
          denom += alpha[O.length-1][i]*beta[O.length-1][i];
      }
      for (int i = 0; i < numberOfStates; ++i) {
        gamma[O.length-1][i] = 0.0;
        gamma[O.length-1][i] += (alpha[O.length-1][i]*beta[O.length-1][i])/denom;
      }
      
      /* Re-estimate A,B and pi */
      
      //Pi
      for (int i = 0; i < numberOfStates; ++i) {
        pi[i] = gamma[0][i];
      }
      
      //A
      for (int i = 0; i < numberOfStates; ++i) {
        for (int j = 0; j < numberOfStates; ++j) {
          numer = 0.0;
          denom = 0.0;
          for (int t = 0; t < O.length-1; ++t) {
            numer += xi[t][i][j];
            denom += gamma[t][i];
          }
          if (denom != 0){
            A[i][j] = numer/denom;
          } else {
            A[i][j] = 0;
          }
            
        }
      }

      //B
      for (int i = 0; i < numberOfStates; ++i) {
        for (int j = 0; j < numberOfEmissions; ++j) {
          numer = 0.0;
          denom = 0.0;
          for (int t = 0; t < O.length; ++t) {
            if (j == O[t]){
              numer += gamma[t][i];
            }
            denom += gamma[t][i];
          }
          if (denom != 0){
            B[i][j] = numer/denom;
          } else {
            B[i][j] = 0;
          }
        }
      }
      
      /* Compute log probability for model generating observed sequence */
      
      logProb = 0.0;
      for (int t = 0; t < C.length; ++t) {
        if (C[t] != 0){
          logProb += Math.log(C[t]);
        }
      }
      logProb = -logProb;
      if (logProb > oldLogProb){
        iters += 1;
        oldLogProb = logProb;
      } else {
        finished = true;
      }
    }
  }

  public double[][] getA(){
    double[][]result = new double[numberOfStates][numberOfStates];
    for (int i = 0; i < numberOfStates; ++i){
      for (int j = 0; j < numberOfStates; ++j){
        result[i][j] = this.A[i][j];
      }
    }
    return result;
 } 

  public double[][] getB(){
    double[][]result = new double[numberOfStates][numberOfEmissions];
    for (int i = 0; i < numberOfStates; ++i){
      for (int j = 0; j < numberOfEmissions; ++j){
        result[i][j] = this.B[i][j];
      }
    }
    return result;
  }

  /**
   * Copies a HMM onto this one.
   * 
   * @param hmm the HMM you are going to copy to this one.
   */
  public void copyHMM(HMM hmm){
    for (int i = 0; i < numberOfStates; ++i){
      for (int j = 0; j < numberOfStates; ++j){
        A[i][j] = hmm.A[i][j];
      }
      pi[i] = hmm.pi[i];
    }
    for (int i = 0; i < numberOfStates; ++i){
      for (int j = 0; j < numberOfEmissions; ++j){
        B[i][j] = hmm.B[i][j];
      }
    }
  }
  
  /**
   * Divides all the entries of A, B and pi by a divisor.
   * 
   * @param divisor the divisor
   */
  public void divide(int divisor) {
    for (int i = 0; i < numberOfStates; i++) {
      for (int j = 0; j < numberOfStates; j++) {
        A[i][j] = A[i][j] / divisor;
      }
    }
    for (int i = 0; i < numberOfStates; i++) {
      for (int j = 0; j < numberOfEmissions; j++) {
        B[i][j] = B[i][j] / divisor;
      }
    }
    for (int i = 0; i < numberOfStates; i++) {
      pi[i] = pi[i] / divisor;
    }
}
  
  /**
   * Multiplies all the entries of A, B and pi with a factor.
   * 
   * @param factor the factor
   */
  public void multiply(int factor) {
    for (int i = 0; i < numberOfStates; i++) {
      for (int j = 0; j < numberOfStates; j++) {
        A[i][j] = A[i][j] * factor;
      }
    }
    for (int i = 0; i < numberOfStates; i++) {
      for (int j = 0; j < numberOfEmissions; j++) {
        B[i][j] = B[i][j] * factor;
      }
    }
    for (int i = 0; i < numberOfStates; i++) {
      pi[i] = pi[i] * factor;
    }
  }
  
  /**
   * Adds all the entries of A, B and pi of a HMM to the corresponding entries
   * of this HMM.
   * 
   * @param hmm the HMM you want to add entrywise to this one.
   */
  public void add(HMM hmm){
    for (int i = 0; i < numberOfStates; ++i){
      for (int j = 0; j < numberOfStates; ++j){
        A[i][j] += hmm.A[i][j];
      }
      pi[i] += hmm.pi[i];
    }
    
    for (int i = 0; i < numberOfStates; ++i){
      for (int j = 0; j < numberOfEmissions; ++j){
        B[i][j] += hmm.B[i][j];
      }
    }
  }

  public void printMatrix(){
    System.err.println("\nA");
    for (int i = 0; i < numberOfStates; ++i) {
      for (int j = 0; j < numberOfStates; ++j) {
                System.err.print(this.A[i][j]+ "  ");
      }

       System.err.println("");
    }

    System.err.println("\nB");
    for (int i = 0; i < numberOfStates; ++i) {
      for (int j = 0; j < numberOfEmissions; ++j) {
                System.err.print(this.B[i][j]+ "  ");
      }

       System.err.println("");
    }

     System.err.println("\nPi");
     for (int i = 0; i < numberOfStates; i++) {
         System.err.print(pi[i]+ "  ");
      }
    System.err.println("");
   


  }

  public static double[][] normalize(double m[][]) {
      double[][] m2 = new double[m.length][m[0].length];

      for (int row = 0; row < m.length; row++) {
        double sum = sum(m[row]);
        if (sum != 0)
          for (int col = 0; col < m[row].length; col++) {
            m2[row][col] = m[row][col] / sum;
          }
      }

      return m2;

}

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

  public static double sum(double[] prob) {
      double sum = 0;
      for (double d : prob)
        sum += d;

      return sum;
    }

  public double[] currentStateDistribution(int times) {
    double[] s = new double[numberOfStates];
    double[] sBefore = new double[numberOfStates];
    sBefore = this.pi;
    for (int k = 0; k < times; k++) {
      for (int i = 0; i < numberOfStates; i++) {
        for (int j = 0; j < numberOfStates; j++) {
          s[i] += this.A[j][i] * sBefore[j];
        }
      }
      sBefore = s;
    }  
//     System.err.println("S");
//     for (int i = 0; i < numberOfStates; i++) {
//         System.err.print(s[i]+ "  ");
//      }
   
    return s;
  }
  
}

