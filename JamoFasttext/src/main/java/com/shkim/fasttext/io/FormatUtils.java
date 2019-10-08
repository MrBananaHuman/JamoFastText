package com.shkim.fasttext.io;

import org.apache.commons.lang.StringUtils;
import org.apache.commons.lang.Validate;

import java.util.Locale;

/**
 * Minor helper to print and format strings
 *
 * Created by @szuev on 13.12.2017.
 */
public class FormatUtils {

    /**
     * Default: should be similar to c++ output for float
     *
     * @param number float
     * @return String
     */
    public static String toString(float number) {
        return toString(number, 5);
    }

    /**
     * Formats a float value in c++ linux style
     *
     * @param number    float
     * @param precision int, positive
     * @return String
     */
    public static String toString(float number, int precision) {
        Validate.isTrue(precision > 0);
        return String.format(Locale.US, "%." + precision + "g", number).replaceFirst("0+($|e)", "$1").replaceFirst("\\.$", "");
    }

    /**
     * Removes '\n' and '\r' from specified line.
     *
     * @param msg String
     * @return String
     */
    public static String toNonHyphenatedLine(String msg) {
        return StringUtils.replaceChars(msg, "\r\n", null);
    }
}
