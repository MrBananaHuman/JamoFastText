package com.shkim.fasttext;

import java.io.IOException;
import java.util.concurrent.ExecutionException;

import com.shkim.fasttext.module.Main;

public class Train{
	public static void main(String[] args) throws IllegalArgumentException, IOException, ExecutionException {
		String[] inputs = {"skipgram", "-input", "data/train_data.txt", "-output", "vector/word_vector", "-minCount", "1", "-minn", "3", "-maxn", "6", "-dim", "3"};
    	Main.train(inputs);
    	String[] input = {"print-ngram", "vector/word_vector.bin", "1921"};
    	Main.printNgrams(input);
	}
}