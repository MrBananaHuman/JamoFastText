package com.shkim.fasttext.module;

import java.io.IOException;
import java.util.Arrays;
import java.util.Map;
import java.util.Objects;
import java.util.function.Consumer;
import java.util.function.DoubleConsumer;
import java.util.function.IntConsumer;

import com.shkim.fasttext.io.FTInputStream;
import com.shkim.fasttext.io.FTOutputStream;

/**
 * Immutable Args object.
 * See:
 * <a href='https://github.com/facebookresearch/fastText/blob/master/src/args.cc'>args.cc</a> and
 * <a href='https://github.com/facebookresearch/fastText/blob/master/src/args.h'>args.h</a>
 * <p>
 * Help-printing has been moved to {@link Main}.
 * The following settings have been excluded from this class (moved to Main):
 * <code>-input</code>
 * <code>-output</code>
 * <code>-pretrainedVectors</code>
 * <code>-verbose</code>
 * <code>-saveOutput</code>
 * <code>-retrain</code>
 * These settings do not reflect the state or processes of {@link FastText fasttext}
 * and needed once only while running app from command line.
 */
public final class Args {
    // basic:
    private ModelName model = ModelName.SG;
    // dictionary:
    private int minCount = 1;
    private int minCountLabel = 0;
    private int wordNgrams = 1;
    private int bucket = 2_000_000;
    private int minn = 6;
    private int maxn = 12;
    private double t = 1e-4;
    private String label = "__label__";
    // training:
    private double lr = 0.025;
    private int lrUpdateRate = 100;
    private int dim = 300;
    private int ws = 5;
    private int epoch = 5;
    private int neg = 5;
    private LossName loss = LossName.NS;
    private int thread = 12;
    // quantization:
    private boolean qout;
    private boolean qnorm;
    private int dsub = 2;
    private int cutoff;

    public ModelName model() {
        return model;
    }

    public LossName loss() {
        return loss;
    }

    public int wordNgrams() {
        return wordNgrams;
    }

    public int bucket() {
        return bucket;
    }

    public int minn() {
        return minn;
    }

    public int maxn() {
        return maxn;
    }

    public int minCount() {
        return minCount;
    }

    public int minCountLabel() {
        return minCountLabel;
    }

    public double samplingThreshold() {
        return t;
    }

    public String label() {
        return label;
    }

    public double lr() {
        return lr;
    }

    public int lrUpdateRate() {
        return lrUpdateRate;
    }

    public int dim() {
        return dim;
    }

    public int ws() {
        return ws;
    }

    public int epoch() {
        return epoch;
    }

    public int neg() {
        return neg;
    }

    public int thread() {
        return thread;
    }

    public boolean qout() {
        return qout;
    }

    public boolean qnorm() {
        return qnorm;
    }

    public int dsub() {
        return dsub;
    }

    public int cutoff() {
        return cutoff;
    }

    /**
     * <pre>{@code
     * void Args::save(std::ostream& out) {
     *  out.write((char*) &(dim), sizeof(int));
     *  out.write((char*) &(ws), sizeof(int));
     *  out.write((char*) &(epoch), sizeof(int));
     *  out.write((char*) &(minCount), sizeof(int));
     *  out.write((char*) &(neg), sizeof(int));
     *  out.write((char*) &(wordNgrams), sizeof(int));
     *  out.write((char*) &(loss), sizeof(loss_name));
     *  out.write((char*) &(model), sizeof(model_name));
     *  out.write((char*) &(bucket), sizeof(int));
     *  out.write((char*) &(minn), sizeof(int));
     *  out.write((char*) &(maxn), sizeof(int));
     *  out.write((char*) &(lrUpdateRate), sizeof(int));
     *  out.write((char*) &(t), sizeof(double));
     * }}</pre>
     *
     * @param out {@link FTOutputStream}
     * @throws IOException if an I/O error occurs
     */
    void save(FTOutputStream out) throws IOException {
        out.writeInt(dim);
        out.writeInt(ws);
        out.writeInt(epoch);
        out.writeInt(minCount);
        out.writeInt(neg);
        out.writeInt(wordNgrams);
        out.writeInt(loss.value);
        out.writeInt(model.value);
        out.writeInt(bucket);
        out.writeInt(minn);
        out.writeInt(maxn);
        out.writeInt(lrUpdateRate);
        out.writeDouble(t);
    }

    /**
     * <pre>{@code void Args::load(std::istream& in) {
     *  in.read((char*) &(dim), sizeof(int));
     *  in.read((char*) &(ws), sizeof(int));
     *  in.read((char*) &(epoch), sizeof(int));
     *  in.read((char*) &(minCount), sizeof(int));
     *  in.read((char*) &(neg), sizeof(int));
     *  in.read((char*) &(wordNgrams), sizeof(int));
     *  in.read((char*) &(loss), sizeof(loss_name));
     *  in.read((char*) &(model), sizeof(model_name));
     *  in.read((char*) &(bucket), sizeof(int));
     *  in.read((char*) &(minn), sizeof(int));
     *  in.read((char*) &(maxn), sizeof(int));
     *  in.read((char*) &(lrUpdateRate), sizeof(int));
     *  in.read((char*) &(t), sizeof(double));
     * }}</pre>
     *
     * @param in {@link FTInputStream}
     * @return new instance of Args
     * @throws IOException if an I/O error occurs
     */
    static Args load(FTInputStream in) throws IOException {
        return new Builder()
                .setDim(in.readInt())
                .setWS(in.readInt())
                .setEpoch(in.readInt())
                .setMinCount(in.readInt())
                .setNeg(in.readInt())
                .setWordNgrams(in.readInt())
                .setLossName(LossName.fromValue(in.readInt()))
                .setModel(ModelName.fromValue(in.readInt()))
                .setBucket(in.readInt())
                .setMinN(in.readInt())
                .setMaxN(in.readInt())
                .setLRUpdateRate(in.readInt())
                .setSamplingThreshold(in.readDouble())
                .build();
    }

    @Override
    public String toString() {
        return String.format("{model=%s" +
                        ", minCount=%d, minCountLabel=%d, wordNgrams=%d, bucket=%d, minn=%d, maxn=%d, t=%s, label='%s'" +
                        ", lr=%s, lrUpdateRate=%d, dim=%d, ws=%d, epoch=%d, neg=%d, loss=%s, thread=%d" +
                        ", qout=%s, qnorm=%s, dsub=%d, cutoff=%d}",
                model,
                minCount, minCountLabel, wordNgrams, bucket, minn, maxn, t, label,
                lr, lrUpdateRate, dim, ws, epoch, neg, loss, thread,
                qout, qnorm, dsub, cutoff);
    }
    public static Args parseArgs(Args.ModelName model, Map<String, String> args) throws IllegalArgumentException {
        Args.Builder builder = new Args.Builder().setModel(model);
        putIntegerArg(args, "-lrUpdateRate", builder::setLRUpdateRate);
        putIntegerArg(args, "-dim", builder::setDim);
        putIntegerArg(args, "-ws", builder::setWS);
        putIntegerArg(args, "-epoch", builder::setEpoch);
        putIntegerArg(args, "-minCount", builder::setMinCount);
        putIntegerArg(args, "-minCountLabel", builder::setMinCountLabel);
        putIntegerArg(args, "-neg", builder::setNeg);
        putIntegerArg(args, "-wordNgrams", builder::setWordNgrams);
        putIntegerArg(args, "-bucket", builder::setBucket);
        putIntegerArg(args, "-minn", builder::setMinN);
        putIntegerArg(args, "-maxn", builder::setMaxN);
        putIntegerArg(args, "-thread", builder::setThread);
        putIntegerArg(args, "-cutoff", builder::setCutOff);
        putIntegerArg(args, "-dsub", builder::setDSub);

        putDoubleArg(args, "-lr", builder::setLR);
        putDoubleArg(args, "-t", builder::setSamplingThreshold);

        putBooleanArg(args, "-qnorm", builder::setQNorm);
        putBooleanArg(args, "-qout", builder::setQOut);

        putStringArg(args, "-label", builder::setLabel);

        if (args.containsKey("-loss")) {
            builder.setLossName(Args.LossName.fromName(args.get("-loss")));
        }

        return builder.build();
    }
    private static void putIntegerArg(Map<String, String> map, String key, IntConsumer setter) {
        if (!map.containsKey(key)) return;
        String value = Objects.requireNonNull(map.get(key), "Null int value for " + key);
        try {
            setter.accept(Integer.parseInt(value));
        } catch (NumberFormatException n) {
        }
    }
    private static void putDoubleArg(Map<String, String> map, String key, DoubleConsumer setter) {
        if (!map.containsKey(key)) return;
        String value = Objects.requireNonNull(map.get(key), "Null double value for " + key);
        try {
            setter.accept(Double.parseDouble(value));
        } catch (NumberFormatException n) {
        }
    }
    private static void putBooleanArg(Map<String, String> map, String key, Consumer<Boolean> setter) {
        if (!map.containsKey(key)) return;
        String value = Objects.requireNonNull(map.get(key), "Null value for " + key);
        setter.accept(Boolean.parseBoolean(value));
    }
    private static void putStringArg(Map<String, String> map, String key, Consumer<String> setter) {
        if (!map.containsKey(key)) return;
        setter.accept(Objects.requireNonNull(map.get(key), "Null value for " + key));
    }

    /**
     * The Class-Builder to make new {@link Args args} object.
     * Must be the only way to achieve new instance of {@link Args args}.
     */
    public static class Builder {
        private Args _args = new Args();

        public Builder copy(Args other) {
            return setModel(other.model)
                    // dictionary:
                    .setLabel(other.label).setWordNgrams(other.wordNgrams)
                    .setMinCount(other.minCount).setMinCountLabel(other.minCountLabel)
                    .setBucket(other.bucket).setSamplingThreshold(other.t)
                    .setMinN(other.minn).setMaxN(other.maxn)
                    // train:
                    .setLossName(other.loss).setDim(other.dim).setWS(other.ws)
                    .setLR(other.lr).setLRUpdateRate(other.lrUpdateRate).setNeg(other.neg)
                    .setEpoch(other.epoch).setThread(other.thread)
                    // quantization:
                    .setQNorm(other.qnorm).setQOut(other.qout).setCutOff(other.cutoff).setDSub(other.dsub);
        }

        public Builder setModel(ModelName name) {
            _args.model = Objects.requireNonNull(name, "Null model name");
            return this;
        }

        public Builder setLossName(LossName name) {
            _args.loss = Objects.requireNonNull(name, "Null loss name");
            return this;
        }

        public Builder setDim(int dim) {
            _args.dim = requirePositive(dim, "dim");
            return this;
        }

        public Builder setWS(int ws) {
            _args.ws = requirePositive(ws, "ws");
            return this;
        }

        public Builder setLR(double lr) {
            _args.lr = requirePositive(lr, "lr");
            return this;
        }

        public Builder setLRUpdateRate(int lrUpdateRate) {
            _args.lrUpdateRate = requirePositive(lrUpdateRate, "lrUpdateRate");
            return this;
        }

        public Builder setWordNgrams(int wordNgrams) {
            _args.wordNgrams = requirePositive(wordNgrams, "wordNgrams");
            return this;
        }

        public Builder setMinCount(int minCount) {
            _args.minCount = requirePositive(minCount, "minCount");
            return this;
        }

        public Builder setMinCountLabel(int minCountLabel) {
            _args.minCountLabel = requireNotNegative(minCountLabel, "minCountLabel");
            return this;
        }

        public Builder setNeg(int neg) {
            _args.neg = requirePositive(neg, "neq");
            return this;
        }

        public Builder setBucket(int bucket) {
            _args.bucket = requireNotNegative(bucket, "bucket");
            return this;
        }

        public Builder setMinN(int minn) {
            _args.minn = requireNotNegative(minn, "minn");
            return this;
        }

        public Builder setMaxN(int maxn) {
            _args.maxn = requireNotNegative(maxn, "maxn");
            return this;
        }

        public Builder setEpoch(int epoch) {
            _args.epoch = requirePositive(epoch, "epoch");
            return this;
        }

        public Builder setThread(int thread) {
            _args.thread = requireNotNegative(thread, "thread");
            return this;
        }

        public Builder setSamplingThreshold(double t) {
            _args.t = requirePositive(t, "samplingThreshold");
            return this;
        }

        public Builder setLabel(String label) {
            _args.label = Objects.requireNonNull(label, "Null label prefix");
            return this;
        }

        public Builder setQNorm(boolean qnorm) {
            _args.qnorm = qnorm;
            return this;
        }

        public Builder setQOut(boolean qout) {
            _args.qout = qout;
            return this;
        }

        public Builder setCutOff(int cutoff) {
            _args.cutoff = requireNotNegative(cutoff, "cutoff");
            return this;
        }

        public Builder setDSub(int dsub) {
            _args.dsub = requirePositive(dsub, "dsub");
            return this;
        }

        public Args build() {
            if (ModelName.SUP.equals(_args.model)) {
                _args.loss = LossName.SOFTMAX;
                _args.minCount = 1;
                _args.minn = 0;
                _args.maxn = 0;
                _args.lr = 0.1;
            }
            if (_args.wordNgrams <= 1 && _args.maxn == 0) {
                _args.bucket = 0;
            }
            return _args;
        }

        private static int requirePositive(int val, String name) {
            if (val > 0) return val;
            throw new IllegalArgumentException("The '" + name + "' must be positive: " + val);
        }

        private static int requireNotNegative(int val, String name) {
            if (val >= 0) return val;
            throw new IllegalArgumentException("The '" + name + "' must not be negative: " + val);
        }

        private static double requirePositive(double val, String name) {
            if (val > 0) return val;
            throw new IllegalArgumentException("The '" + name + "' must be positive: " + val);
        }
    }

    public enum ModelName {
        CBOW("cbow", 1), SG("skipgram", 2), SUP("supervised", 3);

        private final int value;
        private final String name;

        ModelName(String name, int value) {
            this.name = name;
            this.value = value;
        }

        public String getName() {
            return name;
        }

        public static ModelName fromValue(int value) throws IllegalArgumentException {
            return Arrays.stream(values()).filter(v -> v.value == value)
                    .findFirst().orElseThrow(() -> new IllegalArgumentException("Unknown model enum value: " + value));
        }

        public static ModelName fromName(String value) {
            return Arrays.stream(values()).filter(v -> v.name.equalsIgnoreCase(value))
                    .findFirst().orElseThrow(() -> new IllegalArgumentException("Unknown model name: " + value));
        }
    }

    public enum LossName {
        HS(1), NS(2), SOFTMAX(3);
        private final int value;

        LossName(int value) {
            this.value = value;
        }

        public static LossName fromValue(int value) throws IllegalArgumentException {
            return Arrays.stream(values()).filter(v -> v.value == value)
                    .findFirst().orElseThrow(() -> new IllegalArgumentException("Unknown loss enum value: " + value));
        }

        public static LossName fromName(String value) throws IllegalArgumentException {
            return Arrays.stream(values()).filter(v -> v.name().equalsIgnoreCase(value))
                    .findFirst().orElseThrow(() -> new IllegalArgumentException("Unknown loss name: " + value));
        }
    }

}
