package de.unima.ki.anyburl.algorithm;


import java.io.PrintWriter;
// import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;

import de.unima.ki.anyburl.Settings;
import de.unima.ki.anyburl.data.*;
import de.unima.ki.anyburl.structure.*;
import de.unima.ki.anyburl.structure.compare.RuleConfidenceComparator;
import de.unima.ki.anyburl.threads.Predictor;

public class RuleEngine {

	private final static double EPSILON = 0.0001;
	
	
	private static LinkedList<Triple> predictionTasks = new LinkedList<Triple>();
	private static int predictionsMade = 0;
	private static PrintWriter predictionsWriter = null;
	
	private static int DEBUG_TESTSET_SUBSET = 0;
	

	
	
	public static void materializeRules(LinkedList<Rule> rules, TripleSet trainingSet, TripleSet materializedSet) {
		int ruleCounter = 0;
		
		for (Rule rule : rules) {
			ruleCounter++;
			if (ruleCounter % (rules.size() / 100) == 0) System.out.println("* " + (100.0 * (ruleCounter / (double)rules.size())) + "% of all rules materialized");
			if (rule.bodysize() > 2) continue;
			TripleSet materializedRule = rule.materialize(trainingSet);
			if (materializedRule != null) {
				// System.out.println(materializedRule.size());
				materializedSet.addTripleSet(materializedRule);
				// System.out.println(materializedSet.size());
			}
		}
	}
	
	
	public static void applyRulesARX(LinkedList<Rule> rules, TripleSet testSet, TripleSet trainingSet, TripleSet validationSet, int k, PrintWriter resultsWriter) {
		
		if (DEBUG_TESTSET_SUBSET > 0) {
			System.out.println("* debugging mode, choosing small fraction of testset");
			TripleSet testSetReduced = new TripleSet();
			for (int i = 0; i < DEBUG_TESTSET_SUBSET; i++) {
				Triple t = testSet.getTriples().get(i);
				testSetReduced.addTriple(t);
				
				
			}
			for (int i = DEBUG_TESTSET_SUBSET; i < testSet.getTriples().size(); i++) {
				Triple t = testSet.getTriples().get(i);
				validationSet.addTriple(t);
			}
			testSet = testSetReduced;
		}
		
		System.out.println("* applying rules");
		// HashMap<String, HashSet<Rule>> relation2Rules = createRuleIndex(rules);

		HashMap<String, ArrayList<Rule>> relation2Rules4Prediction = createOrderedRuleIndex(rules);
		
		
		
		
		// TODO fix something here 
		System.out.println("* set up index structure covering rules for prediction for " + relation2Rules4Prediction.size() + " relations");
		//TripleSet filterSet = new TripleSet();
		
		// filterSet.addTripleSet(trainingSet);
		// filterSet.addTripleSet(validationSet);
		// filterSet.addTripleSet(testSet);
		// if (materializedSet != null) trainingSet.addTripleSet(materializedSet);
		
	
		
		// System.out.println("* constructed filter set with " + filterSet.getTriples().size() + " triples");
		//if (filterSet.getTriples().size() == 0) {
		//	System.err.println("WARNING: using empty filter set!");
		//}
		// prepare the data structures used a s cache for question that are reoccuring
		// HashMap<SimpleImmutableEntry<String, String>, LinkedHashMap<String, Double>> headCandidateCache = new HashMap<SimpleImmutableEntry<String, String>, LinkedHashMap<String, Double>>();
		// HashMap<SimpleImmutableEntry<String, String>, LinkedHashMap<String, Double>> tailCandidateCache = new HashMap<SimpleImmutableEntry<String, String>, LinkedHashMap<String, Double>>();
		// start iterating over the test cases
		
		// int counter = 0;
		// long startTime = System.currentTimeMillis();
		// long currentTime = 0;
		
		
		ScoreTree.LOWER_BOUND = k;
		ScoreTree.UPPER_BOUND = ScoreTree.LOWER_BOUND;
		ScoreTree.EPSILON = EPSILON;

		
		predictionTasks.addAll(testSet.getTriples());
		predictionsWriter = resultsWriter;
		
		Thread[] predictors = new Thread[Settings.WORKER_THREADS];
		System.out.print("* creating worker threads ");
		for (int threadCounter = 0; threadCounter < Settings.WORKER_THREADS; threadCounter++) {
			System.out.print("#" + threadCounter + " ");
			predictors[threadCounter] = new Predictor(testSet, trainingSet, validationSet, k, relation2Rules4Prediction);
			predictors[threadCounter].start();
		}
		System.out.println();
		
		
		
		while (alive(predictors)) {
			try {
				Thread.sleep(500);
			}
			catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		predictionsWriter.flush();
		predictionsWriter.close();
		
		System.out.println("* done with rule application");
		RuleEngine.predictionsMade = 0;

	}
	
	private static boolean alive(Thread[] threads) {
		for (Thread t : threads) {
			if (t.isAlive()) return true;
		}
		return false;
	}
	
	
	
	public static synchronized Triple getNextPredictionTask()  {
		predictionsMade++;
		Triple triple = predictionTasks.poll();
		if (predictionsMade % 100 == 0) {
			if (triple != null) System.out.println("* (#" + predictionsMade + ") trying to guess the tail and head of " + triple.toString());
			predictionsWriter.flush();
		}
		return triple;
	}
	
	
	/*
	
	public static synchronized Rule getNextRuleMaterializationTask()  {
		Rule rule = ruleMaterializationTasksListed.poll();
		if (rule == null) return null;
		// ruleMaterializationTasks.remove(rule);
		return rule;
	}
	
	public static synchronized void addRuleToBeMaterialized(Rule rule)  {
		if (ruleMaterializationTasks.contains(rule)) return;
		ruleMaterializationTasksListed.add(rule);
		// System.out.println("list=" + ruleMaterializationTasksListed.size());
		ruleMaterializationTasks.add(rule);
		// System.out.println("set=" + ruleMaterializationTasks.size());
	}
	
	public static void materializeRule(Rule rule, TripleSet ts) {

		System.err.println("materialize rule: " + rule);
		TripleSet materializedSet = rule.materialize(ts);
		System.err.println("finished");
		
		synchronized (materializedRules) {
			materializedRules.put(rule, materializedSet);
			ruleMaterializationsMade++;
		}
		System.err.println("... and stored");
		
	}
	
	*/
	
	
	
	public static void predictMax(TripleSet testSet, TripleSet trainingSet, TripleSet validationSet, int k, HashMap<String, ArrayList<Rule>> relation2Rules4Prediction, Triple triple) {
		//System.out.println("=== " + triple + " ===");
		ScoreTree kTree = new ScoreTree();
		LinkedHashMap<String, Double> kTailCandidates = predictMax(testSet, trainingSet, validationSet, k, relation2Rules4Prediction, triple, false, kTree);
		ScoreTree kTailTree = kTree;
		kTree = new ScoreTree();

		LinkedHashMap<String, Double> kHeadCandidates = predictMax(testSet, trainingSet, validationSet, k, relation2Rules4Prediction, triple, true, kTree);
		ScoreTree kHeadTree = kTree;
		
		if (Settings.PATH_EXPLANATION != null) writeTopKExplanation(triple, testSet, kHeadCandidates, kHeadTree, kTailCandidates, kTailTree, k);
		writeTopKCandidates(triple, testSet, kHeadCandidates, kTailCandidates, predictionsWriter, k);
	}
	
	/*
	private static void replaceSelfByValue_(LinkedHashMap<String, Double> kCandidates, String value) {
		
		if (kCandidates.containsKey(Settings.REWRITE_REFLEXIV_TOKEN)) {
			double confidence = kCandidates.get(Settings.REWRITE_REFLEXIV_TOKEN);
			kCandidates.remove(Settings.REWRITE_REFLEXIV_TOKEN);
			kCandidates.put(value, confidence);
		}
	}
	*/

	
	// OLD VERSION: its completely unclear why the previous
	// there is not difference on FB15k between old and new version, thats why new should be preferred


	public static LinkedHashMap<String, Double> predictMaxOLD(TripleSet testSet, TripleSet trainingSet, TripleSet filterSet, int k, HashMap<String, ArrayList<Rule>> relation2Rules, Triple triple, boolean predictHeadNotTail, ScoreTree kTree) {

		String relation = triple.getRelation();
		String head = triple.getHead();
		String tail = triple.getTail();

		if (relation2Rules.containsKey(relation)) {
			ArrayList<Rule> relevantRules = relation2Rules.get(relation);
			
			Rule previousRule = null;
			Set<String> candidates = new HashSet<String>();
			Set<String> fCandidates = new HashSet<String>();
			
			for (Rule rule : relevantRules) {
				// long startTime = System.currentTimeMillis();
				if (previousRule != null) {
					if (predictHeadNotTail) candidates = previousRule.computeHeadResults(tail, trainingSet);
					else candidates = previousRule.computeTailResults(head, trainingSet);
					fCandidates.addAll(getFilteredEntities(trainingSet, filterSet, testSet, triple, candidates, !predictHeadNotTail)); // the negation seems to be okay here
					if (previousRule.getAppliedConfidence() > rule.getAppliedConfidence()) {
						if (!kTree.fine()) {
							if (fCandidates.size() > 0) {
								if (Settings.PATH_EXPLANATION != null) kTree.addValues(previousRule.getAppliedConfidence(), fCandidates, previousRule);
								else kTree.addValues(previousRule.getAppliedConfidence(), fCandidates, null);
								fCandidates.clear();
							}
						}
						else break;
					}
				}
				previousRule = rule;
			}
			
			if (!kTree.fine() && previousRule != null) {
				if (predictHeadNotTail) candidates = previousRule.computeHeadResults(tail, trainingSet);
				else candidates = previousRule.computeTailResults(head, trainingSet);
				fCandidates.addAll(getFilteredEntities(trainingSet, filterSet, testSet, triple, candidates, !predictHeadNotTail));
				if (Settings.PATH_EXPLANATION != null) kTree.addValues(previousRule.getAppliedConfidence(), fCandidates, previousRule);
				else kTree.addValues(previousRule.getAppliedConfidence(), fCandidates, null);
				fCandidates.clear();
			}
		}
		
		LinkedHashMap<String, Double> kCandidates = new LinkedHashMap<String, Double>();		
		kTree.getAsLinkedList(kCandidates, (predictHeadNotTail ? tail : head));
		
		return kCandidates;

	}
	
	
	// NEW SIMPLIFIED VERSION
	public static LinkedHashMap<String, Double> predictMax(TripleSet testSet, TripleSet trainingSet, TripleSet validationSet, int k, HashMap<String, ArrayList<Rule>> relation2Rules, Triple triple, boolean predictHeadNotTail, ScoreTree kTree) {

		String relation = triple.getRelation();
		String head = triple.getHead();
		String tail = triple.getTail();

		if (relation2Rules.containsKey(relation)) {
			ArrayList<Rule> relevantRules = relation2Rules.get(relation);
			Set<String> candidates = new HashSet<String>();
			Set<String> fCandidates = new HashSet<String>();
			for (Rule rule : relevantRules) {
				if (predictHeadNotTail) candidates = rule.computeHeadResults(tail, trainingSet);
				else candidates = rule.computeTailResults(head, trainingSet);
				fCandidates.addAll(getFilteredEntities(trainingSet, validationSet, testSet, triple, candidates, !predictHeadNotTail)); // the negation seems to be okay here
				if (!kTree.fine()) {
					if (fCandidates.size() > 0) {
						if (Settings.PATH_EXPLANATION != null) kTree.addValues(rule.getAppliedConfidence(), fCandidates, rule);
						else kTree.addValues(rule.getAppliedConfidence(), fCandidates, null);
						fCandidates.clear();
					}
				}
				else break;
			}
		}
		LinkedHashMap<String, Double> kCandidates = new LinkedHashMap<String, Double>();		
		kTree.getAsLinkedList(kCandidates, (predictHeadNotTail ? tail : head));
		
		return kCandidates;

	}
	
	


	
	

	public static void predictNoisyOr(TripleSet testSet, TripleSet trainingSet, TripleSet validationSet, int k, HashMap<String, ArrayList<Rule>> relation2Rules, Triple triple) {
		String relation = triple.getRelation();
		String head = triple.getHead();
		String tail = triple.getTail();

		HashMap<String, ArrayList<Rule>> explainedTailCandidates = new HashMap<String, ArrayList<Rule>>();
		HashMap<String, ArrayList<Rule>> explainedHeadCandidates = new HashMap<String, ArrayList<Rule>>();

		if (relation2Rules.containsKey(relation)) {
			ArrayList<Rule> relevantRules = relation2Rules.get(relation);
			for (Rule rule : relevantRules) {

				HashSet<String> tailCandidates = rule.computeTailResults(head, trainingSet);
				HashSet<String> fTailCandidates = getFilteredEntities(trainingSet, validationSet, testSet, triple,
						tailCandidates, true);
				for (String fTailCandidate : fTailCandidates) {
					if (!explainedTailCandidates.containsKey(fTailCandidate))
						explainedTailCandidates.put(fTailCandidate, new ArrayList<Rule>());
					explainedTailCandidates.get(fTailCandidate).add(rule);
				}

				HashSet<String> headCandidates = rule.computeHeadResults(tail, trainingSet);
				HashSet<String> fHeadCandidates = getFilteredEntities(trainingSet, validationSet, testSet, triple,
						headCandidates, false);
				for (String fHeadCandidate : fHeadCandidates) {
					if (!explainedHeadCandidates.containsKey(fHeadCandidate))
						explainedHeadCandidates.put(fHeadCandidate, new ArrayList<Rule>());
					explainedHeadCandidates.get(fHeadCandidate).add(rule);
				}

			}

		}

		LinkedHashMap<String, Double> kTailCandidates = new LinkedHashMap<String, Double>();
		LinkedHashMap<String, Double> kHeadCandidates = new LinkedHashMap<String, Double>();


		replaceMyselfByEntity(kTailCandidates, head);
		replaceMyselfByEntity(kHeadCandidates, tail);

		// final sorting
		sortByValue(kTailCandidates);
		sortByValue(kHeadCandidates);

		// if (Settings.PATH_EXPLANATION != null) writeTopKExplanation(triple, testSet,
		// kHeadCandidates, explainedHeadCandidates, kTailCandidates,
		// explainedTailCandidates, k);
		writeTopKCandidates(triple, testSet, kHeadCandidates, kTailCandidates, predictionsWriter, k);
	}
	

	private static void computeNoisyOr(HashMap<String, ArrayList<Rule>> allCandidates, LinkedHashMap<String, Double> kCandidates) {
		for (String cand : allCandidates.keySet()) {
			double log_prob_sum = 0.0;

			int num_rules;
			if (Settings.AGGREGATION_MAX_NUM_RULES_PER_CANDIDATE < 0 || allCandidates.get(cand).size() < Settings.AGGREGATION_MAX_NUM_RULES_PER_CANDIDATE) {
				num_rules = allCandidates.get(cand).size();
			} else {
				num_rules = Settings.AGGREGATION_MAX_NUM_RULES_PER_CANDIDATE;
			}

			int ctr = 1;
			for (Rule r : allCandidates.get(cand)) {
				log_prob_sum += Math.log(1 - r.getAppliedConfidence());
				if (ctr == num_rules) {
					break;
				} else {
					ctr += 1;
				}
			}
			// double score = 1-Math.exp(log_prob_sum);
			double score = -1 * log_prob_sum;
			kCandidates.put(cand, score);
		}
	}
	
	
	public static void replaceMyselfByEntity(LinkedHashMap<String, Double> candidates, String replacement) {
		if (candidates.containsKey(Settings.REWRITE_REFLEXIV_TOKEN)) {
			double myselfConf = candidates.get(Settings.REWRITE_REFLEXIV_TOKEN);
			candidates.remove(Settings.REWRITE_REFLEXIV_TOKEN);
			candidates.put(replacement, myselfConf);
		}
	}
	

	/*
	private static void show(LinkedHashMap<String, Double> kCandidates, String headline) {
		System.out.println("*** " + headline + " ***");
		for (String candidate : kCandidates.keySet()) {
			double conf = kCandidates.get(candidate);
			System.out.println(conf + " = " + candidate);
		}
	}

	private static HashMap<String, HashSet<Rule>> createRuleIndex(List<Rule> rules) {
		
		// int counterL1C = 0;
		// int counterL2C = 0;
		// int counterL1AC = 0;
		// int counterL1AN = 0;
		// int counterOther = 0;
		
		HashMap<String, HashSet<Rule>> relation2Rules = new HashMap<String, HashSet<Rule>>();
		for (Rule rule : rules) {
			
			
			
			if (rule.isXYRule()) {
				if (rule.bodysize() == 1)  counterL1C++;
				if (rule.bodysize() == 2)  counterL2C++;
			}
			else {
				
				if (rule.bodysize() == 1)  {
					if (rule.hasConstantInBody()) counterL1AC++;
					else counterL1AN++;
				}
				else {
					if (rule.hasConstantInBody()) continue;
				}	
			}
			
			
			String relation = rule.getTargetRelation();
			if (!relation2Rules.containsKey(relation)) relation2Rules.put(relation, new HashSet<Rule>());
			relation2Rules.get(relation).add(rule);
			
			
		}
		// System.out.println("L1C=" + counterL1C + " L2C=" + counterL2C + " L1AC=" + counterL1AC + " L1AN=" + counterL1AN + " OTHER=" + counterOther);
		return relation2Rules;
	}
	*/
	
	
	public static HashMap<String, ArrayList<Rule>> createOrderedRuleIndex(LinkedList<Rule> rules) {
		// String predictionGoal = headNotTailPrediction ? "head" : "tail";
		HashMap<String, ArrayList<Rule>> relation2Rules = new HashMap<String, ArrayList<Rule>>();
		long l = 0;
		for (Rule rule : rules) {

			if (Settings.THRESHOLD_CORRECT_PREDICTIONS > rule.getCorrectlyPredicted()) continue;
			if (Settings.THRESHOLD_CONFIDENCE > rule.getConfidence()) continue;

			String relation = rule.getTargetRelation();
			if (!relation2Rules.containsKey(relation)) {
				relation2Rules.put(relation, new ArrayList<Rule>());
			}
			relation2Rules.get(relation).add(rule);	
			if (l % 100000 == 0 && l > 1) {
				System.out.println("* indexed " + l + " rules for prediction");
			}
			l++;
		}
		for (String relation : relation2Rules.keySet()) {
			relation2Rules.get(relation).trimToSize();
			Collections.sort(relation2Rules.get(relation), new RuleConfidenceComparator());
		}
		System.out.println("* indexed and sorted " + l + " rules for using them to make predictions");
		return relation2Rules;
	}


	/*
	private static void updateCandidateProbabailities(Rule rule, boolean tailNotHead, String candidate, HashMap<String, Double> candidates2Probabilities) {
		double prob = rule.getAppliedConfidence();
		if (!candidates2Probabilities.containsKey(candidate)) candidates2Probabilities.put(candidate, prob);
		else {
			double previousProb = candidates2Probabilities.get(candidate);
			double newProb = combineProbability(prob, previousProb);
			candidates2Probabilities.put(candidate, newProb);
		}
	}
	*/
	
	/*
	private static LinkedHashMapK getFilteredCandidates(TripleSet filterSet, TripleSet testSet, Triple t, HashMap<String, Double> candidates, boolean tailNotHead) {
		// LinkedHashMap<String, Double> candidatesSorted = sortByValue(candidates);
		LinkedHashMap<String, Double> kCandidates = new LinkedHashMap<String, Double>();
		int i = 0;
		for (Entry<String, Double> entry : candidates.entrySet()) {
			if (!tailNotHead) {
				if (!filterSet.isTrue(entry.getKey(), t.getRelation(), t.getTail())) {
					kCandidates.put(entry.getKey(), entry.getValue());
					i++;
				}
				if (testSet.isTrue(entry.getKey(), t.getRelation(), t.getTail())) {
					kCandidates.put(entry.getKey(), entry.getValue());
				}
			}
			if (tailNotHead) {
				if (!filterSet.isTrue(t.getHead(), t.getRelation(), entry.getKey())) {
					kCandidates.put(entry.getKey(), entry.getValue());
					i++;
				}
				if (testSet.isTrue(t.getHead(), t.getRelation(), entry.getKey())) {
					kCandidates.put(entry.getKey(), entry.getValue());
				}
			}
		}
		return (new LinkedHashMapK(kCandidates, i));
	}
	*/
	
	
	
	private static HashSet<String> getFilteredEntities(TripleSet trainingSet, TripleSet validationSet, TripleSet testSet, Triple t, Set<String> candidateEntities, boolean tailNotHead) {
		// LinkedHashMap<String, Double> candidatesSorted = sortByValue(candidates);
		HashSet<String> filteredEntities = new HashSet<String>();
		for (String entity : candidateEntities) {
			if (!tailNotHead) {		
				if (!validationSet.isTrue(entity, t.getRelation(), t.getTail()) && !trainingSet.isTrue(entity, t.getRelation(), t.getTail()) && !testSet.isTrue(entity, t.getRelation(), t.getTail())) {
					filteredEntities.add(entity);
				}
				if (testSet.isTrue(entity, t.getRelation(), t.getTail())) {
					// TAKE CARE, remove to reactivate the possibility of storing previous results
					if (entity.equals(t.getHead())) filteredEntities.add(entity);
				}	
			}
			if (tailNotHead) {
				if (!validationSet.isTrue(t.getHead(), t.getRelation(), entity) && !trainingSet.isTrue(t.getHead(), t.getRelation(), entity) && !testSet.isTrue(t.getHead(), t.getRelation(), entity)) {
					filteredEntities.add(entity);
				}
				if (testSet.isTrue(t.getHead(), t.getRelation(), entity)) {
					// TAKE CARE, remove to reactivate the possibility of storing previous results
					if (entity.equals(t.getTail())) filteredEntities.add(entity);
				}
			}
		}
		return filteredEntities;
	}
	
	private static synchronized void writeTopKCandidates(Triple t, TripleSet testSet, LinkedHashMap<String, Double> kHeadCandidates, LinkedHashMap<String, Double> kTailCandidates, PrintWriter writer, int k) {
		writer.println(t);
		int i = 0;
		writer.print("Heads: ");
		for (Entry<String, Double> entry : kHeadCandidates.entrySet()) {
			if (t.getHead().equals(entry.getKey()) || !testSet.isTrue(entry.getKey(), t.getRelation(), t.getTail())) {
				writer.print(entry.getKey() + "\t" + entry.getValue() + "\t");
				i++;
			}
			if (i == k) break;
		}
		writer.println();
		i = 0;
		writer.print("Tails: ");
		for (Entry<String, Double> entry : kTailCandidates.entrySet()) {
			if (t.getTail().equals(entry.getKey()) || !testSet.isTrue(t.getHead(), t.getRelation(), entry.getKey())) {
				writer.print(entry.getKey() + "\t" + entry.getValue() + "\t");
				i++;
			}
			if (i == k) break;
		}
		writer.println();
		writer.flush();

	}
	
	private static synchronized void writeTopKExplanation(Triple t, TripleSet testSet, LinkedHashMap<String, Double> kHeadCandidates, ScoreTree headTree, LinkedHashMap<String, Double> kTailCandidates, ScoreTree tailTree, int k) {
		Settings.EXPLANATION_WRITER.println(t);
		Settings.EXPLANATION_WRITER.println("Heads:");
		Settings.EXPLANATION_WRITER.println(headTree);
		Settings.EXPLANATION_WRITER.println("Tails:");
		Settings.EXPLANATION_WRITER.println(tailTree);
		Settings.EXPLANATION_WRITER.flush();
	}
	
	/*
	private static synchronized void writeTopKCandidatesPlusExplanation(Triple t, TripleSet testSet, LinkedHashMap<String, Double> kHeadCandidates, ScoreTree allHeadCandidates, LinkedHashMap<String, Double> kTailCandidates, ScoreTree allTailCandidates,	PrintWriter writer, int k) {
		Settings.EXPLANATION_WRITER.println(t);
		Settings.EXPLANATION_WRITER.println("Heads:");
		Settings.EXPLANATION_WRITER.println(allHeadCandidates);
		Settings.EXPLANATION_WRITER.println("Tails:");
		Settings.EXPLANATION_WRITER.println(allTailCandidates);
		Settings.EXPLANATION_WRITER.flush();
	}
	*/
	
	
	/*
	private static void processTopKCandidates(TripleSet testSet, Triple t, HashMap<String, Double> tailCandidates, HashMap<String, Double> headCandidates, TripleSet filterSet, int k, PrintWriter writer, HashMap<String, Double> kTailCandidates, HashMap<String, Double> kHeadCandidates) {
		LinkedHashMap<String, Double> tailCandidatesSorted = sortByValue(tailCandidates);
		LinkedHashMap<String, Double> headCandidatesSorted = sortByValue(headCandidates);
		writer.println(t);
		writer.print("Heads: ");
		int i = 0;
		for (Entry<String, Double> entry : headCandidatesSorted.entrySet()) {
			if (i < k) {
				if (!filterSet.isTrue(entry.getKey(), t.getRelation(), t.getTail()) || t.getHead().equals(entry.getKey())) {
					writer.print(entry.getKey() + "\t" + entry.getValue() + "\t");
					kHeadCandidates.put(entry.getKey(), entry.getValue());
					i++;
				}
				if (testSet.isTrue(entry.getKey(), t.getRelation(), t.getTail())) {
					kHeadCandidates.put(entry.getKey(), entry.getValue());
				}
			}
		}
		writer.println();
		writer.print("Tails: ");
		int j = 0;
		for (Entry<String, Double> entry : tailCandidatesSorted.entrySet()) {
			if (j < k) {
				if (!filterSet.isTrue(t.getHead(), t.getRelation(), entry.getKey())
						|| t.getTail().equals(entry.getKey())) {
					writer.print(entry.getKey() + "\t" + entry.getValue() + "\t");
					kTailCandidates.put(entry.getKey(), entry.getValue());
					j++;
				}
				if (testSet.isTrue(t.getHead(), t.getRelation(), entry.getKey())) {
					kTailCandidates.put(entry.getKey(), entry.getValue());
				}
			}
		}
		writer.println();
		writer.flush();
	}
	*/

	/*
	private static double combineProbability(double prob, double previousProb) {
		double newProb;
		switch (COMBINATION_RULE_ID) {
		case 1: // multiplication
			newProb = 1.0 - ((1.0 - previousProb) * (1.0 - prob));
			break;
		case 2: // maxplus
			newProb = Math.max(previousProb, prob) + EPSILON;
			break;
		case 3: // max
		default:
			newProb = Math.max(previousProb, prob);
			break;
		}
		return newProb;
	}
	*/
	

	/*
	private static <K, V extends Comparable<? super V>> LinkedHashMap<K, V> sortByValue(Map<K, V> map) {
		return map.entrySet().stream().sorted(Map.Entry.comparingByValue(Collections.reverseOrder()))
				.collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue, (e1, e2) -> e1, LinkedHashMap::new));
	}
	*/
	
	public static void sortByValue(LinkedHashMap<String, Double> m) {
	    List<Map.Entry<String, Double>> entries = new ArrayList<>(m.entrySet());
	    Collections.sort(entries, new Comparator<Map.Entry<String, Double>>() {
	        public int compare(Map.Entry<String, Double> lhs, Map.Entry<String, Double> rhs) {
	            if (lhs.getValue() < rhs.getValue()) return 1;
	            else if (lhs.getValue() > rhs.getValue())  return -1;
	            else return 0;
	        }
	    });

	    m.clear();
	    for(Map.Entry<String, Double> e : entries) {
	        m.put(e.getKey(), e.getValue());
	    }
	}
	
	
}
