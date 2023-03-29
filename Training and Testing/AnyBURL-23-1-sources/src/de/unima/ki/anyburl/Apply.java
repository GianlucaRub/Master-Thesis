package de.unima.ki.anyburl;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.LinkedList;
import java.util.Properties;

import de.unima.ki.anyburl.algorithm.RuleEngine;
import de.unima.ki.anyburl.data.Triple;
import de.unima.ki.anyburl.data.TripleSet;
import de.unima.ki.anyburl.io.IOHelper;
import de.unima.ki.anyburl.io.RuleReader;
import de.unima.ki.anyburl.structure.Rule;
import de.unima.ki.anyburl.structure.RuleAcyclic1;
import de.unima.ki.anyburl.structure.RuleAcyclic2;
import de.unima.ki.anyburl.structure.RuleCyclic;

public class Apply {
	
	

		
	
	private static String CONFIG_FILE = "config-apply.properties";
	

	private static String PW_JOINT_FILE = "";
	
	/**
	 * Filter the rule set prior to applying it. Removes redundant rules which do not have any impact (or no desired impact). 
	 */
	// public static boolean FILTER = true;	

	
	
	/**
	* Always should be set to false. The TILDE results are based on a setting where this is set to true.
	* This parameter is sued to check in how far this setting increases the quality of the results.
	*/
	public static boolean USE_VALIDATION_AS_BK = false;


	
	
	public static void main(String[] args) throws IOException {
		
		if (args.length == 1) {
			CONFIG_FILE = args[0];
			System.out.println("* reading params from file " + CONFIG_FILE);
		}
		
		if (args.length == 2) {
			CONFIG_FILE = args[0];
			PW_JOINT_FILE = args[1];
			System.out.println("* reading params from file " + CONFIG_FILE);
			System.out.println("* using pairwise joint file file " + PW_JOINT_FILE);
		}
		
		
		
		Rule.applicationMode();
		Properties prop = new Properties();
		InputStream input = null;
		try {
			input = new FileInputStream(CONFIG_FILE);
			prop.load(input);
			Settings.PREDICTION_TYPE = IOHelper.getProperty(prop, "PREDICTION_TYPE", Settings.PREDICTION_TYPE);
			if (Settings.PREDICTION_TYPE.equals("aRx")) {
				Settings.SAFE_PREFIX_MODE = IOHelper.getProperty(prop, "SAFE_PREFIX_MODE", Settings.SAFE_PREFIX_MODE);
				Settings.PATH_TRAINING = IOHelper.getProperty(prop, "PATH_TRAINING", Settings.PATH_TRAINING);
				Settings.PATH_TEST = IOHelper.getProperty(prop, "PATH_TEST", Settings.PATH_TEST);
				Settings.PATH_VALID = IOHelper.getProperty(prop, "PATH_VALID",Settings.PATH_VALID);
				Settings.PATH_OUTPUT = IOHelper.getProperty(prop, "PATH_OUTPUT", Settings.PATH_OUTPUT);
				Settings.PATH_EXPLANATION = IOHelper.getProperty(prop, "PATH_EXPLANATION", Settings.PATH_EXPLANATION);
				Settings.PATH_RULES = IOHelper.getProperty(prop, "PATH_RULES", Settings.PATH_RULES);
				Settings.PATH_RULES_BASE = IOHelper.getProperty(prop, "PATH_RULES_BASE", Settings.PATH_RULES_BASE);
				Settings.TOP_K_OUTPUT = IOHelper.getProperty(prop, "TOP_K_OUTPUT", Settings.TOP_K_OUTPUT);
				Settings.UNSEEN_NEGATIVE_EXAMPLES = IOHelper.getProperty(prop, "UNSEEN_NEGATIVE_EXAMPLES", Settings.UNSEEN_NEGATIVE_EXAMPLES);
				Settings.UNSEEN_NEGATIVE_EXAMPLES_REFINE = IOHelper.getProperty(prop, "UNSEEN_NEGATIVE_EXAMPLES_REFINE", Settings.UNSEEN_NEGATIVE_EXAMPLES_REFINE);
				Settings.THRESHOLD_CONFIDENCE = IOHelper.getProperty(prop, "THRESHOLD_CONFIDENCE", Settings.THRESHOLD_CONFIDENCE);
				Settings.DISCRIMINATION_BOUND = IOHelper.getProperty(prop, "DISCRIMINATION_BOUND",  Settings.DISCRIMINATION_BOUND);
				Settings.TRIAL_SIZE = IOHelper.getProperty(prop, "TRIAL_SIZE",  Settings.TRIAL_SIZE);
				Settings.WORKER_THREADS = IOHelper.getProperty(prop, "WORKER_THREADS",  Settings.WORKER_THREADS);
				Settings.BEAM_SAMPLING_MAX_BODY_GROUNDINGS = IOHelper.getProperty(prop, "BEAM_SAMPLING_MAX_BODY_GROUNDINGS", Settings.BEAM_SAMPLING_MAX_BODY_GROUNDINGS);
				Settings.BEAM_SAMPLING_MAX_BODY_GROUNDING_ATTEMPTS = IOHelper.getProperty(prop, "BEAM_SAMPLING_MAX_BODY_GROUNDING_ATTEMPTS", Settings.BEAM_SAMPLING_MAX_BODY_GROUNDING_ATTEMPTS);
				Settings.BEAM_SAMPLING_MAX_REPETITIONS = IOHelper.getProperty(prop, "BEAM_SAMPLING_MAX_REPETITIONS", Settings.BEAM_SAMPLING_MAX_REPETITIONS);
				
				Settings.READ_CYCLIC_RULES   = IOHelper.getProperty(prop, "READ_CYCLIC_RULES",  Settings.READ_CYCLIC_RULES);
				Settings.READ_ACYCLIC1_RULES = IOHelper.getProperty(prop, "READ_ACYCLIC1_RULES",  Settings.READ_ACYCLIC1_RULES);
				Settings.READ_ACYCLIC2_RULES = IOHelper.getProperty(prop, "READ_ACYCLIC2_RULES",  Settings.READ_ACYCLIC2_RULES);
				Settings.READ_ZERO_RULES = IOHelper.getProperty(prop, "READ_ZERO_RULES",  Settings.READ_ZERO_RULES);
				
				Settings.READ_THRESHOLD_CONFIDENCE = IOHelper.getProperty(prop, "READ_THRESHOLD_CONFIDENCE", Settings.READ_THRESHOLD_CONFIDENCE);
				Settings.READ_THRESHOLD_CORRECT_PREDICTIONS = IOHelper.getProperty(prop, "READ_THRESHOLD_CORRECT_PREDICTIONS", Settings.READ_THRESHOLD_CORRECT_PREDICTIONS);
				Settings.READ_THRESHOLD_MAX_RULE_LENGTH = IOHelper.getProperty(prop, "READ_THRESHOLD_MAX_RULE_LENGTH", Settings.READ_THRESHOLD_MAX_RULE_LENGTH);
				
				//FILTER = IOHelper.getProperty(prop, "FILTER", FILTER);
				Settings.AGGREGATION_TYPE = IOHelper.getProperty(prop, "AGGREGATION_TYPE", Settings.AGGREGATION_TYPE);
			
				Settings.RULE_AC2_WEIGHT = IOHelper.getProperty(prop, "RULE_AC2_WEIGHT", Settings.RULE_AC2_WEIGHT);
				Settings.RULE_ZERO_WEIGHT = IOHelper.getProperty(prop, "RULE_ZERO_WEIGHT", Settings.RULE_ZERO_WEIGHT);
				// Settings.RULE_LENGTH_DEGRADE = IOHelper.getProperty(prop, "RULE_LENGTH_DEGRADE", Settings.RULE_LENGTH_DEGRADE);
				
				Settings.REWRITE_REFLEXIV = false;
				
				
				if (Settings.AGGREGATION_TYPE.equals("maxplus")) Settings.AGGREGATION_ID = 1;
				if (Settings.AGGREGATION_TYPE.equals("max2")) Settings.AGGREGATION_ID = 2;
				if (Settings.AGGREGATION_TYPE.equals("noisyor")) Settings.AGGREGATION_ID = 3;
				if (Settings.AGGREGATION_TYPE.equals("maxgroup")) Settings.AGGREGATION_ID = 4;
			}
			else {
				System.err.println("The prediction type " + Settings.PREDICTION_TYPE + " is not yet supported.");
				System.exit(1);
			}

		}
		catch (IOException ex) {
			System.err.println("Could not read relevant parameters from the config file " + CONFIG_FILE);
			ex.printStackTrace();
			System.exit(1);
		}
		
		finally {
			if (input != null) {
				try {
					input.close();
				}
				catch (IOException e) {
					e.printStackTrace();
					System.exit(1);
				}
			}
		}
		
		if (Settings.PREDICTION_TYPE.equals("aRx")) {
				
			String[] values = getMultiProcessing(Settings.PATH_RULES);
			
			PrintWriter log = null;
			
			if (values.length == 0) log = new PrintWriter(Settings.PATH_RULES + "_plog");
			else log = new PrintWriter(Settings.PATH_OUTPUT.replace("|", "") + "_plog");
			
			log.println("Logfile");
			log.println("~~~~~~~\n");
			log.println();
			log.println(IOHelper.getParams());
			log.flush();
			
			RuleReader rr = new RuleReader();
			LinkedList<Rule> base = new LinkedList<Rule>();
			
			
			if (!Settings.PATH_RULES_BASE.equals("")) {
				System.out.println("* reading additional rule file as base");
				base = rr.read(Settings.PATH_RULES_BASE);
			}
			
			
			
			for (String value : values) {
				
				long startTime = System.currentTimeMillis();
				
				// long indexStartTime = System.currentTimeMillis();
				String path_output_used = null;
				String path_rules_used = null;
				if (value == null) {
					path_output_used = Settings.PATH_OUTPUT;
					path_rules_used = Settings.PATH_RULES;
				}
				if (value != null) {
					path_output_used = Settings.PATH_OUTPUT.replaceFirst("\\|.*\\|", "" + value); 
					path_rules_used = Settings.PATH_RULES.replaceFirst("\\|.*\\|", "" + value);
					
				}
				log.println("rules:   " + path_rules_used);
				log.println("output: " + path_output_used);
				log.flush();
			
				PrintWriter pw = new PrintWriter(new File(path_output_used));

				
				if (Settings.PATH_EXPLANATION != null) Settings.EXPLANATION_WRITER = new PrintWriter(new File(Settings.PATH_EXPLANATION));
				System.out.println("* writing prediction to " + path_output_used);
				
				TripleSet trainingSet = new TripleSet(Settings.PATH_TRAINING);
				
				TripleSet testSet = new TripleSet(Settings.PATH_TEST);
				TripleSet validSet = new TripleSet(Settings.PATH_VALID);
				
				// check if you should predict only unconnected
				// ACHTUNG: Never remove that comment here
				// checkIfPredictOnlyUnconnected(validSet, trainingSet);
				
				if (USE_VALIDATION_AS_BK) {
					trainingSet.addTripleSet(validSet);
					validSet = new TripleSet();
				}
				//DecimalFormat df = new DecimalFormat("0.0000");
				//System.out.println("MEMORY REQUIRED (before reading rules): " + df.format(Runtime.getRuntime().totalMemory() / 1000000.0) + " MByte");
				
				
				LinkedList<Rule> rules = rr.read(path_rules_used);

				rules.addAll(base);
				
				//System.out.println("MEMORY REQUIRED (after reading rules): " + df.format(Runtime.getRuntime().totalMemory() / 1000000.0) + " MByte");
				

				int rulesSize = rules.size();
				LinkedList<Rule> rulesThresholded = new LinkedList<Rule>();
				if (Settings.THRESHOLD_CONFIDENCE > 0.0) {
					for (Rule r : rules) {
						
						// if (r instanceof RuleAcyclic1 && (r.bodysize() == 3 || r.bodysize() == 2) && r.getHead().getConstant().equals(r.getBodyAtom(r.bodysize()-1).getConstant())) continue;
						if (r.getConfidence() > Settings.THRESHOLD_CONFIDENCE) {
							rulesThresholded.add(r);	
						}
					}
					System.out.println("* applied confidence threshold of " + Settings.THRESHOLD_CONFIDENCE + " and reduced from " + rules.size() + " to " + rulesThresholded.size() + " rules");
				}
				rules = rulesThresholded;
				
				long startApplicationTime = System.currentTimeMillis();
				RuleEngine.applyRulesARX(rules, testSet, trainingSet, validSet, Settings.TOP_K_OUTPUT, pw);
			
			
				long endTime = System.currentTimeMillis();
				System.out.println("* evaluated " + rulesSize + " rules to propose candiates for " + testSet.getTriples().size() + "*2 completion tasks");
				System.out.println("* finished in " + (endTime - startTime) + "ms.");
				
				
				System.out.println();
				
				
				long indexEndTime = System.currentTimeMillis();
				
				log.println("finished in " + (endTime - startApplicationTime) / 1000 + "s (rule indexing and application, creation and storage of ranking).");
				log.println("finished in " + (endTime - startTime) / 1000 + "s including all operations (+ loading triplesets,  + loading rules).");
				log.println();
				log.flush();
			}
			log.close();

		}	
	}




	private static void filterTSA(String[] TSA, int TSAindex, LinkedList<Rule> rulesThresholded, Rule r) {
		switch(TSA[TSAindex]) {
		case "ALL":
			rulesThresholded.add(r);
			break;
		case "C-1":
			if (r instanceof RuleCyclic && r.bodysize() == 1) rulesThresholded.add(r);
			break;
		case "C-2":
			if (r instanceof RuleCyclic && r.bodysize() == 2) rulesThresholded.add(r);
			break;
		case "C-3":
			if (r instanceof RuleCyclic && r.bodysize() == 3) rulesThresholded.add(r);
			break;
		case "AC1-1":
			if (r instanceof RuleAcyclic1 && r.bodysize() == 1) rulesThresholded.add(r);
			break;
		case "AC1-2":
			if (r instanceof RuleAcyclic1 && r.bodysize() == 2) rulesThresholded.add(r);
			break;	
		case "AC2-1":
			if (r instanceof RuleAcyclic2 && r.bodysize() == 1) rulesThresholded.add(r);
			break;
		case "AC2-2":
			if (r instanceof RuleAcyclic2 && r.bodysize() == 2) rulesThresholded.add(r);
			break;	
		case "N-C-1":
			if (!(r instanceof RuleCyclic && r.bodysize() == 1)) rulesThresholded.add(r);
			break;
		case "N-C-2":
			if (!(r instanceof RuleCyclic && r.bodysize() == 2)) rulesThresholded.add(r);
			break;
		case "N-C-3":
			if (!(r instanceof RuleCyclic && r.bodysize() == 3)) rulesThresholded.add(r);
			break;
		case "N-AC1-1":
			if (!(r instanceof RuleAcyclic1 && r.bodysize() == 1)) rulesThresholded.add(r);
			break;
		case "N-AC1-2":
			if (!(r instanceof RuleAcyclic1 && r.bodysize() == 2)) rulesThresholded.add(r);
			break;	
		case "N-AC2-1":
			if (!(r instanceof RuleAcyclic2 && r.bodysize() == 1)) rulesThresholded.add(r);
			break;
		case "N-AC2-2":
			if (!(r instanceof RuleAcyclic2 && r.bodysize() == 2)) rulesThresholded.add(r);
			break;		
		}
	}
	
	

	/*
	private static void showRulesStats(List<Rule> rules) { 
		int xyCounter = 0;
		int xCounter = 0;
		int yCounter = 0;
		for (Rule rule : rules) {
			if (rule.isXYRule()) xyCounter++;
			if (rule.isXRule()) xCounter++;
			if (rule.isYRule()) yCounter++;
		}
		System.out.println("XY=" + xyCounter + " X="+ xCounter + " Y=" + yCounter);
		
	}
	*/

	public static String[] getMultiProcessing(String path1) {
		String token[] = path1.split("\\|");
		if (token.length < 2) {
			return new String[] {null};
		}
		else {
			String values[] = token[1].split(",");
			return values;
		}
	}


	

		

}
