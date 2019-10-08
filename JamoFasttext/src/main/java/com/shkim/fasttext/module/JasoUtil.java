package com.shkim.fasttext.module;

import java.util.HashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 한글의 자소를 다루는 클래스
 * 
 * @author kblee
 *
 */
public final class JasoUtil {
  
  /**
   * Constructor
   */
  JasoUtil() {
    
  }
  
  static final char[] ChoSung = { 0x3131, 0x3132, 0x3134, 0x3137, 0x3138, 0x3139, 0x3141, 0x3142, 0x3143, 0x3145,
      0x3146, 0x3147, 0x3148, 0x3149, 0x314a, 0x314b, 0x314c, 0x314d, 0x314e };
  // ㅏ ㅐ ㅑ ㅒ ㅓ ㅔ ㅕ ㅖ ㅗ ㅘ ㅙ ㅚ ㅛ ㅜ ㅝ ㅞ ㅟ ㅠ ㅡ ㅢ ㅣ
  static final char[] JwungSung = { 0x314f, 0x3150, 0x3151, 0x3152, 0x3153, 0x3154, 0x3155, 0x3156, 0x3157, 0x3158,
      0x3159, 0x315a, 0x315b, 0x315c, 0x315d, 0x315e, 0x315f, 0x3160, 0x3161, 0x3162, 0x3163 };
  // ㄱ ㄲ ㄳ ㄴ ㄵ ㄶ ㄷ ㄹ ㄺ ㄻ ㄼ ㄽ ㄾ ㄿ ㅀ ㅁ ㅂ ㅄ ㅅ ㅆ ㅇ ㅈ ㅊ ㅋ ㅌ ㅍ ㅎ
  static final char[] JongSung = { 0, 0x3131, 0x3132, 0x3133, 0x3134, 0x3135, 0x3136, 0x3137, 0x3139, 0x313a, 0x313b,
      0x313c, 0x313d, 0x313e, 0x313f, 0x3140, 0x3141, 0x3142, 0x3144, 0x3145, 0x3146, 0x3147, 0x3148, 0x314a, 0x314b,
      0x314c, 0x314d, 0x314e };
  
  private static final char[] tmpCho = { 'ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 
      'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ' };
  private static final char[] tmpJung = { 'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 
      'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ' };
  private static final char[] tmpJong = { ' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 
      'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ' };
  private static final String[] doubleConsonant = 
      { "ㄱㅅ", "ㄴㅈ", "ㄴㅎ", "ㄹㄱ", "ㄹㅁ", "ㄹㅂ", "ㄹㅅ", "ㄹㅌ", "ㄹㅎ", "ㅂㅅ" };
  private static final String[] coupleConsonant = 
      { "ㄳ", "ㄵ", "ㄶ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㅀ", "ㅄ" };
  private static final String[] doubleVowel = 
      { "ㅗㅏ", "ㅗㅐ", "ㅗㅣ", "ㅜㅓ", "ㅜㅔ", "ㅜㅣ", "ㅡㅣ" };
  private static final String[] coupleVowel = { "ㅘ", "ㅙ", "ㅚ", "ㅝ", "ㅞ", "ㅟ", "ㅢ" };
  
  /**
   * 초성
   */
  private static final Map<Character, Integer> choHan = new HashMap<>();
  
  static {
    for (int i = 0; i < tmpCho.length; i++) {
      choHan.put(tmpCho[i], i);
    }
  }
  
  /**
   * 중성
   */
  private static final Map<Character, Integer> jungHan = new HashMap<>();
  
  static {
    for (int i = 0; i < tmpJung.length; i++) {
      jungHan.put(tmpJung[i], i);
    }
  }
  
  /**
   * 종성
   */
  private static final Map<Character, Integer> jongHan = new HashMap<>();
  
  static {
    for (int i = 0; i < tmpJong.length; i++) {
      jongHan.put(tmpJong[i], i);
    }
  }
  
  /**
   * coupleJongsung
   */
  private static final Map<String, String> coupleJongsung = new HashMap<>();
  
  static {
    for (int i = 0; i < doubleConsonant.length; i++) {
      coupleJongsung.put(doubleConsonant[i], coupleConsonant[i]);
    }
  }
  
  /**
   * coupleJungsung
   */
  private static final Map<String, String> coupleJungsung = new HashMap<>();
  
  static {
    for (int i = 0; i < doubleVowel.length; i++) {
      coupleJungsung.put(doubleVowel[i], coupleVowel[i]);
    }
  }
  
  private static final char uniCodeHBase = 0xAC00;
  private static final int CONSONANT_COUNT = 3;
  private static final int VOWELS_COUNT = 2;
  private static final int CHOSUNG_VALUE = 21;
  private static final int JUNGSUNG_VALUE = 28;

  private static final int JUNG_SIZE = 21;
  private static final int JONG_SIZE = 28;
  
  // * 한글 초성
  private static final char[] initialConsonant = { 'ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ',
      'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ' };
  // * 한글 중성
  private static final char[] medialVowel = { 'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ',
      'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ' };
  // * 한글 종성
  private static final char[] finalConsonant = { ' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ',
      'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ' };
  
  // 초성
  public static char getChosung(char ch) {
    return initialConsonant[(ch - 0xAC00) / (JUNG_SIZE * JONG_SIZE)];
  }

  // 중성
  public static char getJungsung(char ch) {
    return medialVowel[((ch - 0xAC00) % (JONG_SIZE * JUNG_SIZE)) / JONG_SIZE];
  }

  // 종성
  public static char getJongsung(char ch) {
    return finalConsonant[(ch - 0xAC00) % JONG_SIZE];
  }
  
  /**
   * 자음과 모음의 집합중에 겹자음이나 겹모음이 될 수 있는지 확인하고, 변형해 주는 함수
   * @param jamos
   * @return 겹자음 및 겹모음으로 변형된 문자열
   */
  public static String replaceSingleToCouple(String jamos) {
    
    Pattern pattern = Pattern.compile("([ㄱ-ㅎ]+|[ㅏ-ㅣ]+)");
    Matcher matcher = pattern.matcher(jamos);
    
    String replaceJamo = jamos;
    while (matcher.find()) {
      if (matcher.group(0).length() >= CONSONANT_COUNT && matcher.group(0).matches("[ㄱ-ㅎ]+")) {
        String replacement = matcher.group(0);
        
        int conCount = 0;
        while (conCount < (matcher.group(0).length() - 2)) {
          String part = matcher.group(0).substring(conCount,conCount + 2);
          if (coupleJongsung.containsKey(part)) {
            replacement = replacement.replaceFirst(part, coupleJongsung.get(part));
            conCount += 2;
          } else {
            conCount += 1;
          }
        }
        replaceJamo = replaceJamo.replaceFirst(matcher.group(0), replacement);
      } else if (matcher.group(0).length() >= VOWELS_COUNT && matcher.group(0).matches("[ㅏ-ㅣ]+")) {
        String replacement = matcher.group(0);
        
        int voCount = 0;
        while (voCount < matcher.group(0).length() - 1) {
          String part = matcher.group(0).substring(voCount,voCount + 2);
          if (coupleJungsung.containsKey(part)) {
            replacement = replacement.replaceFirst(part, coupleJungsung.get(part));
            voCount += 2;
          } else {
            voCount += 1;
          }
        }
        replaceJamo = replaceJamo.replaceFirst(matcher.group(0), replacement);
      }
    }
    return replaceJamo;
  }
  
  /**
   * 한글을 입력 받아 자소를 분리해줍니다.
   * @param text
   * @return 분리된 자소로 이루어진 문자열
   */
  public static String hangulToJaso(String text) {
    // 유니코드 한글 문자열을 입력 받음
    int cho;
    int jung;
    int jong; // 자소 버퍼: 초성/중성/종성 순
    
    StringBuilder result = new StringBuilder();
    
    result.setLength(0);

    for (int i = 0; i < text.length(); i++) {
      char ch = text.charAt(i);

      if (ch >= uniCodeHBase && ch <= 0xD7A3) { // "AC00:가" ~ "D7A3:힣" 에 속한 글자면
        // 분해

        jong = ch - uniCodeHBase;
        cho = jong / (CHOSUNG_VALUE * JUNGSUNG_VALUE);
        jong = jong % (CHOSUNG_VALUE * JUNGSUNG_VALUE);
        jung = jong / JUNGSUNG_VALUE;
        jong = jong % JUNGSUNG_VALUE;

        result.append(ChoSung[cho]);
        result.append(JwungSung[jung]);
        if (jong != 0) {
          // c가 0이 아니면, 즉 받침이 있으면
          result.append(JongSung[jong]);
          result.append("ᴥ");
        } else {
        	result.append("ᴥ");
        }
      } else {
        result.append(ch);
        
      }
    }
    return result.toString();
  }
  
  
  public static String hangulToJaso_with_symbol(String text) {
	    // 유니코드 한글 문자열을 입력 받음
	    int cho;
	    int jung;
	    int jong; // 자소 버퍼: 초성/중성/종성 순
	    
	    StringBuilder result = new StringBuilder();
	    
	    result.setLength(0);

	    for (int i = 0; i < text.length(); i++) {
	      char ch = text.charAt(i);

	      if (ch >= uniCodeHBase && ch <= 0xD7A3) { // "AC00:가" ~ "D7A3:힣" 에 속한 글자면
	        // 분해

	        jong = ch - uniCodeHBase;
	        cho = jong / (CHOSUNG_VALUE * JUNGSUNG_VALUE);
	        jong = jong % (CHOSUNG_VALUE * JUNGSUNG_VALUE);
	        jung = jong / JUNGSUNG_VALUE;
	        jong = jong % JUNGSUNG_VALUE;

	        result.append(ChoSung[cho]);
	        result.append(JwungSung[jung]);
	        if (jong != 0) {
	          // c가 0이 아니면, 즉 받침이 있으면
	          result.append(JongSung[jong]);
	          result.append("ᴥ");
	        } else {
	        	result.append("ᴥ");
	        }
	      } else {
	        result.append(ch);
	        
	      }
	    }
	    return result.toString();
	  }
  /**
   * 자소들로 이루어진 문자열을 입력받아, 조합하는 함수입니다.
   * 문자열을 뒤에서 부터 읽어 중성을 기준으로 조합 합니다.
   * @param jamos
   * @return 조합이 완료된 문자열
   */
  public static String jasoToHangul(String jamos) {
    
    String replacedJamos = replaceSingleToCouple(jamos);
    StringBuilder result = new StringBuilder(replacedJamos);
    
    int iterate = replacedJamos.length() - 1;
    int currentWorkIndex = replacedJamos.length();
    // 조합을 완료한 Index
    
    while (iterate > 0) {
      char currentJamo = replacedJamos.charAt(iterate);
      if (!jungHan.containsKey(currentJamo)) {
        iterate--;
        continue;
      }
      
      char preJamo = replacedJamos.charAt(iterate - 1);
      char afterJamo = ' ';
      int afterJamoIndex = iterate + 1;
      
      if ( afterJamoIndex < replacedJamos.length() && afterJamoIndex < currentWorkIndex) {
        // 종성 여부 확인
        afterJamo = replacedJamos.charAt(afterJamoIndex);
        if (!jongHan.containsKey(afterJamo)) {
          afterJamo = ' ';
        }
      }
      
      boolean choJung = choHan.containsKey(preJamo) && jungHan.containsKey(currentJamo);
      
      if (choJung && (!jongHan.containsKey(afterJamo) || afterJamo == ' ')) {
        // 초성 및 중성의 조합의 경우
        String replacement = String.valueOf((char) (uniCodeHBase
            + (choHan.get(preJamo) * CHOSUNG_VALUE + jungHan.get(currentJamo)) * JUNGSUNG_VALUE + 0));
        result.replace(iterate - 1, iterate + 1, replacement);
        iterate--;
        currentWorkIndex = iterate;
      } else if (choJung && jongHan.containsKey(afterJamo)) {
        // 초성 중성 종성 모두 조합될 경우
        String replacement = String.valueOf(
            (char) (uniCodeHBase + (choHan.get(preJamo) * CHOSUNG_VALUE + jungHan.get(currentJamo)) * JUNGSUNG_VALUE
                + jongHan.get(afterJamo)));
        result.replace(iterate - 1, iterate + 2, replacement);
        iterate--;
        currentWorkIndex = iterate;
      }
      iterate--;
    }
    return result.toString();
  }
  
  /**
   * reBuildHangulToHangul
   * @param text
   * @return
   */
  public static String reBuildHangulToHangul(String text) {
    return jasoToHangul(hangulToJaso(text));
  }
  
  
  public static void main(String[] args) {
	  String aa = "에피타이저	애메랄드	애스키모	오셰아니아	인스턴트	카타고리	컨설탄트	코메디언	페노라마	휴매니스트";
	  System.out.println(hangulToJaso(aa));
	  String input = "ㅅㅓㄱᴥㅆㅣᴥㅇㅝㄴᴥㄹㅠᴥㅇㅡㅇᴥㅎㅘᴥㅅㅏᴥㅈㅓㄱᴥ 0.85381\n" + 
	  		"ㅇㅓㄺᴥㅎㅣᴥㄱㅗᴥㅅㅓㄹᴥㅋㅣㄴᴥ 0.84493\n" + 
	  		"ㅅㅏㅇᴥㅌㅜᴥㅂㅏᴥㅇㅟᴥㄱㅗㄹᴥ 0.84848\n" + 
	  		"ㅇㅜㄹᴥㄹㅜㄱᴥㅂㅜㄹᴥㄹㅜㄱᴥ 0.84646\n" + 
	  		"ㄱㅠᴥㅁㅗᴥㅂㅕㄹᴥ 0.85618\n" + 
	  		"ㅇㅗㄱᴥㅌㅜㅇᴥㅅㅗᴥ 0.84619\n" + 
	  		"ㄱㅏᴥㅉㅏᴥㅅㅣㄴᴥㅅㅓㄴᴥㅌㅏᴥㄹㅕㅇᴥ 0.84661\n" + 
	  		"ㅁㅗㅇᴥㅇㅑㅁᴥㄴㅏᴥ 0.86881\n" + 
	  		"ㄱㅔᴥㅅㅡㅁᴥㅊㅡᴥㄹㅔᴥ 0.84326\n" + 
	  		"ㅇㅓᴥㄷㅜᴥㅋㅓㅁᴥㅋㅓㅁᴥ 0.8455\n" + 
	  		"";
	  input = input.replace("ᴥ", "");
	System.out.println(jasoToHangul(input));
}
}