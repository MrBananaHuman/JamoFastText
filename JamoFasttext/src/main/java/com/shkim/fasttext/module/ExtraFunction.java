package com.shkim.fasttext.module;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

import java.util.concurrent.ExecutionException;

import org.apache.commons.io.IOUtils;

import com.google.common.collect.Multimap;
import com.google.common.math.Stats;

public class ExtraFunction {

	public static InputStream list2stream(List<String> input) throws IOException {
		ByteArrayOutputStream baos = new ByteArrayOutputStream();
		for (String line : input) {
			baos.write(line.getBytes());
		}
		byte[] bytes = baos.toByteArray();
		InputStream in = new ByteArrayInputStream(bytes);
		return in;
	}

	public static File stream2file(InputStream in) throws IOException {
		final File tempFile = File.createTempFile("stream2file", ".tmp");
		tempFile.deleteOnExit();
		try (FileOutputStream out = new FileOutputStream(tempFile)) {
			IOUtils.copy(in, out);
		}
		return tempFile;
	}

	public double[] list2double(List<Float> floatList) {
		double[] doubleArray = floatList.stream().mapToDouble(f -> f != null ? f : Float.NaN) // // want.
				.toArray();
		return doubleArray;
	}

	public double dotProduct(double[] preVector, double[] postVector) {
		double sum = 0;
		for (int i = 0; i < preVector.length; i++) {
			sum += preVector[i] * postVector[i];
		}
		return sum;
	}

	public double dotProduct(Vector preVector, Vector postVector) {
		double sum = 0;
		List<Float> preVectorData = preVector.getData();
		List<Float> postVectorData = postVector.getData();

		double[] preDoubleVector = list2double(preVectorData);
		double[] postDoubleVector = list2double(postVectorData);

		for (int i = 0; i < preDoubleVector.length; i++) {
			sum += preDoubleVector[i] * postDoubleVector[i];
		}
		return sum;
	}
	
	public double dotProductDouble(double[] preDoubleVector, double[] postDoubleVector) {
		double sum = 0;
		for (int i = 0; i < preDoubleVector.length; i++) {
			sum += preDoubleVector[i] * postDoubleVector[i];
		}
		return sum;
	}
	
	public static double norm(double[] array) { 
		 
        if (array != null) { 
            int n = array.length; 
            double sum = 0.0; 
 
            for (int i = 0; i < n; i++) { 
                sum += array[i] * array[i]; 
            } 
            return Math.pow(sum, 0.5); 
        } else 
            return 0.0; 
    }

	public FastText trainFasttextVector(Args args, List<String> trainString)
			throws IOException, IllegalArgumentException, ExecutionException {
		InputStream in = list2stream(trainString);
		File inputFile = stream2file(in);
		FastText vec = FastText.train(args, inputFile.getPath());
		inputFile.delete();
		return vec;
	}

	public FastText trainFasttextVector(Args args, String filePath)
			throws IOException, IllegalArgumentException, ExecutionException {
		FastText vec = FastText.train(args, filePath);
		return vec;
	}

	public void saveFasttextVector(FastText model, String modelPath) throws IllegalArgumentException, IOException {
		model.saveModel(modelPath);
	}

	public FastText loadFastTextVector(String modelPath) throws IllegalArgumentException, IOException {
		FastText vec = FastText.load(modelPath);
		return vec;
	}

	public double[] getFastTextWordVector(String word, FastText vector) {
		Vector vec = vector.getWordVector(word);
		List<Float> floatList = vec.getData();
		double[] doubleArray = list2double(floatList);
		return doubleArray;
	}

	public double[][] getFastTextWordVector(List<String> words, FastText vector) {
		int FastTextize = vector.getArgs().dim();
		int sampleSize = words.size();
		double[][] FastText = new double[sampleSize][FastTextize];

		for (int i = 0; i < sampleSize; i++) {
			String word = words.get(i);
			Vector vec = vector.getWordVector(word);
			List<Float> floatList = vec.getData();
			double[] doubleArray = list2double(floatList);
			for (int k = 0; k < FastTextize; k++) {
				FastText[i][k] = doubleArray[k];
			}
		}

		return FastText;
	}

	public double[][] getFastTextWordVector(String[] words, FastText vector) {
		int FastTextize = vector.getArgs().dim();
		int sampleSize = words.length;
		double[][] FastText = new double[sampleSize][FastTextize];

		for (int i = 0; i < sampleSize; i++) {
			String word = words[i];
			Vector vec = vector.getWordVector(word);
			List<Float> floatList = vec.getData();
			double[] doubleArray = list2double(floatList);
			for (int k = 0; k < FastTextize; k++) {
				FastText[i][k] = doubleArray[k];
			}
		}

		return FastText;
	}

	public double[] getFastTextSentenceVector(String sent, FastText vector) throws IOException {
		Vector vec = vector.getSentenceVector(sent);
		List<Float> floatList = vec.getData();
		double[] doubleArray = list2double(floatList);
		return doubleArray;
	}

	public double[][] getFastTextSentenceVector(List<String> sentences, FastText vector) throws IOException {
		int FastTextize = vector.getArgs().dim();
		int sampleSize = sentences.size();
		double[][] FastText = new double[sampleSize][FastTextize];

		for (int i = 0; i < sampleSize; i++) {
			String sent = sentences.get(i);
			Vector vec = vector.getSentenceVector(sent);
			List<Float> floatList = vec.getData();
			double[] doubleArray = list2double(floatList);
			for (int k = 0; k < FastTextize; k++) {
				FastText[i][k] = doubleArray[k];
			}
		}

		return FastText;
	}

	public double[][] getFastTextSentenceVector(String[] sentences, FastText vector) throws IOException {
		int FastTextize = vector.getArgs().dim();
		int sampleSize = sentences.length;
		double[][] FastText = new double[sampleSize][FastTextize];

		for (int i = 0; i < sampleSize; i++) {
			String sent = sentences[i];
			Vector vec = vector.getSentenceVector(sent);
			List<Float> floatList = vec.getData();
			double[] doubleArray = list2double(floatList);
			for (int k = 0; k < FastTextize; k++) {
				FastText[i][k] = doubleArray[k];
			}
		}

		return FastText;
	}

	public double getFastTextWordSimilarity(String preWord, String postWord, FastText vector) {
		Vector preWordVector = vector.getWordVector(preWord);
		Vector postWordVector = vector.getWordVector(postWord);

		float preWordNorm = preWordVector.norm();
		float postWordNorm = postWordVector.norm();
		double productVector = dotProduct(preWordVector, postWordVector);

		double similarity = productVector / (preWordNorm * postWordNorm);

		return similarity;
	}
	
	public double getVectorSimilarity(double[] preWordVector, double[] postWordVector) {

	    double dotProduct = dotProductDouble(preWordVector, postWordVector);
	    double normA = norm(preWordVector);
	    double normB = norm(postWordVector);
  
	    return dotProduct / (normA * normB);
	}
	

	public double getFastTextSentenceSimilarity(String preSent, String postSent, FastText vector) throws IOException {
		Vector preWordVector = vector.getSentenceVector(preSent);
		Vector postWordVector = vector.getSentenceVector(postSent);

		float preWordNorm = preWordVector.norm();
		float postWordNorm = postWordVector.norm();
		double productVector = dotProduct(preWordVector, postWordVector);

		double similarity = productVector / (preWordNorm * postWordNorm);

		return similarity;
	}

	public double getFastTextHwangSentenceSimilarity(String preSent, String postSent, FastText vector)
			throws IOException {

		String[] preSentWords = preSent.split(" ");
		String[] postSentWords = postSent.split(" ");
		List<Double> allSims = new ArrayList<Double>();

		for (int i = 0; i < preSentWords.length; i++) {
			double maxSim = 0;
			String preWord = preSentWords[i];
			for (int j = 0; j < postSentWords.length; j++) {
				String postWord = postSentWords[j];
				double currentSim = getFastTextWordSimilarity(preWord, postWord, vector);
				if (currentSim > maxSim) {
					maxSim = currentSim;
				}
			}
			allSims.add(maxSim);
		}

		for (int i = 0; i < postSentWords.length; i++) {
			double maxSim = 0;
			String postWord = postSentWords[i];
			for (int j = 0; j < preSentWords.length; j++) {
				String preWord = preSentWords[j];
				double currentSim = getFastTextWordSimilarity(preWord, postWord, vector);
				if (currentSim > maxSim) {
					maxSim = currentSim;
				}
			}
			allSims.add(maxSim);
		}
		double averagedSim = Stats.meanOf(allSims);

		return averagedSim;
	}
	
	public Multimap<String, Float> getMostSimilarWords(String word, int listNum, FastText model){

		int ori_word_id = model.getDictionary().getId(word);
		if(ori_word_id < 0) {
			System.out.println("\tOOV word");
		}
		Multimap<String, Float> results = model.nn(listNum, word);
		
		return results;
	}



}