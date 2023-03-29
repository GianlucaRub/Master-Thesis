package de.unima.ki.anyburl.threads;

import java.util.ArrayList;
import java.util.HashMap;

import de.unima.ki.anyburl.Settings;
import de.unima.ki.anyburl.algorithm.RuleEngine;
import de.unima.ki.anyburl.data.Triple;
import de.unima.ki.anyburl.data.TripleSet;
import de.unima.ki.anyburl.structure.Rule;

/**
 * The predictor predicts the candidates for a knowledge base completion task.
 * 
 *
 */
public class Predictor extends Thread {
	
	private TripleSet testSet;
	private TripleSet trainingSet;
	private TripleSet validationSet;
	private int k;
	private HashMap<String, ArrayList<Rule>> relation2Rules4Prediction;
	
	
	public Predictor(TripleSet testSet, TripleSet trainingSet, TripleSet validationSet, int k, HashMap<String, ArrayList<Rule>> relation2Rules4Prediction) {
		this.testSet = testSet;
		this.trainingSet = trainingSet;
		this.validationSet = validationSet;
		this.k = k;
		this.relation2Rules4Prediction = relation2Rules4Prediction;
	}
	
	
	public void run() {
		Triple triple = RuleEngine.getNextPredictionTask();
		// Rule rule = null;
		while (triple != null) {
			// System.out.println(this.getName() + " making prediction for " + triple);
			if (Settings.AGGREGATION_ID == 1) {
				RuleEngine.predictMax(testSet, trainingSet, validationSet, k, relation2Rules4Prediction, triple);
			}
			triple = RuleEngine.getNextPredictionTask();
		}
		
	}

}
