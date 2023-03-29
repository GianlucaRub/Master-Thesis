package de.unima.ki.anyburl.playground;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.util.Scanner;

import de.unima.ki.anyburl.io.RuleReader;
import de.unima.ki.anyburl.structure.Rule;

/**
 * Coverts the output of AMIE into the rule format of AnyBURL.
 * @author Christian
 *
 */
public class AmieConverter {
	
	public static String[] vars = new String[]{"X", "A", "B", "C", "D", "E"};

	

	
	public static void main(String[] args) throws IOException {
		
		
		
		
		String inputPath  = "exp/large-final/amie/amie-amf-yago-maxad4-rules.txt";
		String outputPath = "exp/large-final/amie/amie-abf-yago-maxad4-rules-new-xxx.txt";

		
		PrintWriter pw = new PrintWriter(outputPath);
		
		RuleReader rr = new RuleReader();
		
		// ?a  hasChild  ?p  ?b  isMarriedTo  ?a  ?a  isMarriedTo  ?p   => ?a  hasChild  ?b	0.011650869	0.776315789	0.867647059	59	76	68	?b
		// ?p  hasChild  ?b  ?b  isMarriedTo  ?a  ?b  isMarriedTo  ?p   => ?a  hasChild  ?b	0.011650869	0.776315789	0.776315789	59	76	76	?b
		
		// Rule r = convertLine(rr, "?a  hasChild  ?p  ?b  isMarriedTo  ?a  ?a  isMarriedTo  ?p   => ?a  hasChild  ?b	0.011650869	0.776315789	0.867647059	59	76	68	?b");
		// System.out.println(r);
		//System.exit(0);
		
		// Rule r = convertLine(rr, "?g  dealsWith  ?a  ?g  dealsWith  ?b   => ?a  dealsWith  ?b	0.475422427	0.193196005	0.229769859	619	3204	2694	?a");
		//Rule r = convertLine(rr, "?b  dealsWith  ?a  ?a  hasNeighbor  ?b   => ?a  dealsWith  ?b	0.041474654	0.327272727	0.350649351	54	165	154	?a");
		
		
		// System.out.println(r);
		
		convertFile(rr, pw, inputPath);
		
		//Rule r = convertLine(rr, "?h  livesIn  ?b  ?h  hasAcademicAdvisor  ?a   => ?a  isCitizenOf  ?b	0.041474654	0.327272727	0.350649351	54	165	154	?a");
	
		
		//System.out.println("Extracted rule: " +  r);

	}
	
	public static void convertFile(RuleReader rr, PrintWriter pw, String inputPath) throws IOException {
		
		
		File file = new File(inputPath);
		BufferedReader br = new BufferedReader((new InputStreamReader(new FileInputStream(file), StandardCharsets.UTF_8)));
		
		int validRuleCounter = 0;
		int invalidRuleCounter = 0;
		
		
		
		try {
		    String line = br.readLine();
		    while (line != null) {
		        if (line == null || line.equals("")) break;
		        if (line.startsWith("?")) { 
		        	Rule r = convertLine(rr, line);
		        	
		        	if (r == null) {
		        		invalidRuleCounter++;
		        		// System.out.println(line);
		        	}
		        	else {
		        		pw.write(r.toString() + "\n");
		        		validRuleCounter++;
		        	}
		        }
		        line = br.readLine();
		    }
		}
		finally {  br.close(); }
		pw.flush();
		pw.close();
		System.out.println("converted amie rule file, valid rules = " + validRuleCounter + ", invalid rules = " + invalidRuleCounter + ".");
	}
	
	
	public static Rule convertLine(RuleReader rr, String line) {

		StringBuilder sb = new StringBuilder("");
		// System.out.println(line);
		String[] parts = line.split("\t");
		String[] hb = parts[0].split("   => ");
		String body = hb[0];
		String head = hb[1];
		
		int correctlyPredicted = Integer.parseInt(parts[4]);
		int predicted = Integer.parseInt(parts[5]);		
		
		sb.append(predicted + "\t");
		sb.append(correctlyPredicted + "\t");
		sb.append(((double)correctlyPredicted / (double)predicted) + "\t");
		
		
		String[] headToken = head.split("  ");
		String xVar = headToken[0];
		String targetRelation = headToken[1];
		String yVar = headToken[2];
		

		sb.append(targetRelation + "(X,Y) <= ");
		// System.out.println(targetRelation + "(X,Y) <=");
		
		
		String[] bodyToken = body.split("  ");

		String v = xVar;
		String xc = "none";
		int counter = 0;
		
		// System.out.println("xVar = " + xVar + "|yVar=" + yVar + "|");
		
		while (!v.equals(yVar)) {
			// System.out.println("v = " + v);
			int ni = getIndexOfExcluded(v, xc, bodyToken);
			if (ni == -1) return null;
			// System.out.println("ni = " + ni);
			int indexAtom = ni / 3;
			int indexWithinAtom = ni % 3;
			int otherIndexWithinAtom = (indexWithinAtom == 0 ? 2 : 0);
			int no = otherIndexWithinAtom + indexAtom * 3;
			int relationIndex = indexAtom * 3 + 1;
			
			String relation = bodyToken[relationIndex];
			xc = v;
			v = bodyToken[no];
			
			String v1 = vars[counter];
			String v2 = vars[counter+1];
			if (v.equals(yVar)) v2 = "Y";
			if (indexWithinAtom == 2) {
				String tmp = v1;
				v1 = v2;
				v2 = tmp;
			}
			
			sb.append(relation + "(" + v1 + "," + v2 + ")");
			if (!v.equals(yVar)) sb.append(", ");

			counter++;
	
		}
		
		//System.out.println(sb.toString());
		
		//System.out.println("bodyToken: " + bodyToken.length);
		//System.out.println("counter: " + counter);

		if (bodyToken.length / 3 == counter) {
			Rule r = rr.getRule(sb.toString());
			return r;
		}
		else {
			return null;
		}

	}

	private static int getIndexOfExcluded(String var, String excluded, String[] token) {
		for (int i = 0; i < token.length; i++) {
			if (token[i].equals(var)) {
				int atomIndex = i / 3;
				int indexWithinAtom = i % 3;
				int otherIndexWithinAtom = atomIndex * 3 + (indexWithinAtom == 0 ? 2 : 0);
				if (!token[otherIndexWithinAtom].equals(excluded)) return  i;
			}
		}
		return -1;
	}
	
	public static  void read() {
		 Scanner scan = new Scanner(System.in);
	     scan.nextLine();
	}


	
}
