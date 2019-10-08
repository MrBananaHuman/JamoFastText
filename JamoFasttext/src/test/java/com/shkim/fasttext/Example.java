package com.shkim.fasttext;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;

import com.google.common.collect.Multimap;
import com.shkim.fasttext.module.Args;
import com.shkim.fasttext.module.ExtraFunction;
import com.shkim.fasttext.module.FastText;


public class Example {
	public static void main(String[] args) throws IllegalArgumentException, IOException, ExecutionException {
		
		ExtraFunction function = new ExtraFunction();
//		Args.ModelName type = Args.ModelName.SG;
//		Map<String, String> modelArgMap = new HashMap<String, String>();
//		 
//		modelArgMap.put("-dim", "10");
//		modelArgMap.put("-minCount", "1");
//		modelArgMap.put("-minn", "2");
//		modelArgMap.put("-maxn", "5");
//		modelArgMap.put("-ws", "5");
//		 
//		Args modelArgs = Args.parseArgs(type, modelArgMap);
//		
//		FastText model = FastText.trainFasttextVector(modelArgs, "data/train_data.txt");
//		
//		model.saveFasttextVector("vector/vector_model");
//		
		FastText trainedModel = FastText.loadFastTextVector("vector/vector_model");
		
		String word1 = "Orange";
		String word2 = "문재인";
		
		Multimap<String, Float> similarWords = trainedModel.getMostSimilarWords(word1, 10);
		System.out.println(word1 + " - sim words: " + similarWords);
		similarWords = trainedModel.getMostSimilarWords(word2, 10);
		System.out.println(word2 + " - sim words: " + similarWords + "\n");
	
		double sim = trainedModel.getFastTextWordSimilarity(word1, word2);
		System.out.println(word1 + " vs " + word2 + " - word sim: " + sim);
		
		double[] word1Vector = trainedModel.getFastTextWordVector(word1);
		double[] word2Vector = trainedModel.getFastTextWordVector(word2);
		
		sim = function.getVectorSimilarity(word1Vector, word2Vector);
		System.out.println(word1 + " vs " + word2 + " - word sim: " + sim + "\n");
		
		String sent1 = "이명박은 대통령이다.";
		String sent2 = "문재인은 대통령이다.";
		//<이명박은>
		sim = trainedModel.getFastTextHwangSentenceSimilarity(sent1, sent2);
		System.out.println("Hwang sent sim: " + sim);
		sim = trainedModel.getFastTextSentenceSimilarity(sent1, sent2);
		System.out.println("Fasttext sent sim: " + sim);
		
		double[] sent1Vector = trainedModel.getFastTextSentenceVector(sent1);
		double[] sent2Vector = trainedModel.getFastTextSentenceVector(sent2);
		
		sim = function.getVectorSimilarity(sent1Vector, sent2Vector);
		System.out.println("Vector sent sim: " + sim);
 
	}
}