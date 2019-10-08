package com.shkim.fasttext.module;

import com.google.common.collect.TreeMultimap;
import com.google.common.primitives.Ints;
import com.google.common.util.concurrent.AtomicDouble;
import com.shkim.fasttext.module.Args.LossName;
import com.shkim.fasttext.module.Args.ModelName;

import org.apache.commons.lang.Validate;
import org.apache.commons.math3.random.RandomAdaptor;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.util.FastMath;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * see <a href='https://github.com/facebookresearch/fastText/blob/master/src/model.cc'>model.cc</a> and
 * <a href='https://github.com/facebookresearch/fastText/blob/master/src/model.h'>model.h</>
 */
public class Model {

    private static final int SIGMOID_TABLE_SIZE = 512;
    private static final int MAX_SIGMOID = 8;
    private static final int LOG_TABLE_SIZE = 512;
    private static final int NEGATIVE_TABLE_SIZE = 10_000_000;

    private static final Comparator<Float> HEAP_PROBABILITY_COMPARATOR = Comparator.reverseOrder();
    // the following order does not important, it is just to match c++ and java versions:
    private static final Comparator<Integer> HEAP_LABEL_COMPARATOR = Comparator.reverseOrder();//Integer::compareTo;

    private static final int PARALLEL_SIZE_THRESHOLD = Integer.parseInt(System.getProperty("parallel.model.threshold",
            String.valueOf(FastText.PARALLEL_THRESHOLD_FACTOR * 100)));

    private QMatrix qwi_;
    private QMatrix qwo_;
    private RandomGenerator rng;
    private Matrix wi_; // input
    private Matrix wo_; // output

    private Vector hidden_;
    private Vector output_;
    private Vector grad_;
    private int osz_; // output vocabSize
    private float loss_;
    private long nexamples_;
    private float[] t_sigmoid; // orig: std::vector<real> t_sigmoid_;
    private float[] t_log; // orig: std::vector<real> t_log_;
    // used for negative sampling:
    private List<Integer> negatives;
    private int negpos;
    // used for hierarchical softmax:
    private List<List<Integer>> paths;
    private List<List<Boolean>> codes;
    private List<Node> tree;

    // args:
    private final Args.ModelName model;
    private final Args.LossName loss;
    private final int dim;
    private final int neg;
    private final boolean qout;

    public Model(Matrix wi, Matrix wo, Args args, RandomGenerator random) {
        this(wi, wo, args.model(), args.loss(), args.dim(), args.neg(), args.qout(), random);
    }

    private Model(Matrix wi, Matrix wo, Args.ModelName model, Args.LossName loss, int dim, int neg, boolean qout, RandomGenerator random) {
        this.model = model;
        this.loss = loss;
        this.dim = dim;
        this.neg = neg;
        this.qout = qout;
        hidden_ = new Vector(dim);
        output_ = new Vector(wo.getM());
        grad_ = new Vector(dim);
        rng = random;
        wi_ = wi;
        wo_ = wo;
        osz_ = wo.getM();
        negpos = 0;
        loss_ = 0.0f;
        nexamples_ = 1L;
        initSigmoid();
        initLog();
    }

    /**
     * <pre>{@code
     * void Model::setQuantizePointer(std::shared_ptr<QMatrix> qwi, std::shared_ptr<QMatrix> qwo, bool qout) {
     *  qwi_ = qwi;
     *  qwo_ = qwo;
     *  if (qout) {
     *      osz_ = qwo_->getM();
     *  }
     * }}</pre>
     *
     * @param qwi {@link QMatrix}, input
     * @param qwo {@link QMatrix}, output
     * @return this model instance
     */
    Model setQuantizePointer(QMatrix qwi, QMatrix qwo) {
        this.qwi_ = qwi;
        this.qwo_ = qwo;
        if (this.qout) {
            this.osz_ = this.qwo_.getM();
        }
        return this;
    }

    RandomGenerator random() {
        return rng;
    }

    public Matrix input() {
        return wi_;
    }

    public Matrix output() {
        return wo_;
    }

    public QMatrix qinput() {
        return qwi_;
    }

    public QMatrix qoutput() {
        return qwo_;
    }

    public boolean isQuant() {
        return qwi_ != null && !qwi_.isEmpty();
    }

    /**
     * <pre>{@code real Model::binaryLogistic(int32_t target, bool label, real lr) {
     *  real score = sigmoid(wo_->dotRow(hidden_, target));
     *  real alpha = lr * (real(label) - score);
     *  grad_.addRow(*wo_, target, alpha);
     *  wo_->addRow(hidden_, target, alpha);
     *  if (label) {
     *      return -log(score);
     *  } else {
     *      return -log(1.0 - score);
     *  }
     * }}</pre>
     *
     * @param target int32_t, index
     * @param label  bool
     * @param lr     float
     * @return float
     */
    private float binaryLogistic(int target, boolean label, float lr) {
        float score = sigmoid(wo_.dotRow(hidden_, target));
        float alpha = lr * ((label ? 1 : 0) - score);
        grad_.addRow(wo_, target, alpha);
        wo_.addRow(hidden_, target, alpha);
        if (label) {
            return -log(score);
        } else {
            return -log(1 - score);
        }
    }

    /**
     * <pre>{@code real Model::negativeSampling(int32_t target, real lr) {
     *  real loss = 0.0;
     *  grad_.zero();
     *  for (int32_t n = 0; n <= args_->neg; n++) {
     *      if (n == 0) {
     *          loss += binaryLogistic(target, true, lr);
     *      } else {
     *          loss += binaryLogistic(getNegative(target), false, lr);
     *      }
     *  }
     *  return loss;
     * }}</pre>
     *
     * @param target int32_t
     * @param lr     float
     * @return float
     */
    private float negativeSampling(int target, float lr) {
        grad_.clear();
        float loss = 0;
        for (int n = 0; n <= neg; n++) {
            if (n == 0) {
                loss += binaryLogistic(target, true, lr);
            } else {
                loss += binaryLogistic(getNegative(target), false, lr);
            }
        }
        return loss;
    }

    /**
     * <pre>{@code real Model::hierarchicalSoftmax(int32_t target, real lr) {
     *  real loss = 0.0;
     *  grad_.zero();
     *  const std::vector<bool>& binaryCode = codes[target];
     *  const std::vector<int32_t>& pathToRoot = paths[target];
     *  for (int32_t i = 0; i < pathToRoot.size(); i++) {
     *      loss += binaryLogistic(pathToRoot[i], binaryCode[i], lr);
     *  }
     *  return loss;
     * }}</pre>
     *
     * @param target int32_t
     * @param lr     float
     * @return float
     */
    private float hierarchicalSoftmax(int target, float lr) {
        float loss = 0.0f;
        grad_.clear();
        List<Boolean> binaryCode = codes.get(target);
        List<Integer> pathToRoot = paths.get(target);
        for (int i = 0; i < pathToRoot.size(); i++) {
            loss += binaryLogistic(pathToRoot.get(i), binaryCode.get(i), lr);
        }
        return loss;
    }

    /**
     * <pre>{@code void Model::computeOutputSoftmax(Vector& hidden, Vector& output) const {
     *  if (quant_ && args_->qout) {
     *      output.mul(*qwo_, hidden);
     *  } else {
     *      output.mul(*wo_, hidden);
     *  }
     *  real max = output[0], z = 0.0;
     *  for (int32_t i = 0; i < osz_; i++) {
     *      max = std::max(output[i], max);
     *  }
     *  for (int32_t i = 0; i < osz_; i++) {
     *      output[i] = exp(output[i] - max);
     *      z += output[i];
     *  }
     *  for (int32_t i = 0; i < osz_; i++) {
     *      output[i] /= z;
     *  }
     * }}</pre>
     *
     * @param hidden {@link Vector}
     * @param output {@link Vector}
     */
    private void computeOutputSoftmax(Vector hidden, Vector output) {
        if (isQuant() && qout) {
            output.mul(qwo_, hidden);
        } else {
            output.mul(wo_, hidden);
        }
        if (FastText.USE_PARALLEL_COMPUTATION && osz_ > PARALLEL_SIZE_THRESHOLD) {
            double max = IntStream.range(0, osz_).parallel().mapToDouble(output::get).max().orElseThrow(() -> new IllegalStateException("Can't calc max"));
            AtomicDouble z = new AtomicDouble();
            IntStream.range(0, osz_).parallel().forEach(i -> {
                double v = FastMath.exp(output.get(i) - max);
                output.set(i, (float) v);
                z.addAndGet(v);
            });
            IntStream.range(0, osz_).parallel().forEach(i -> output.set(i, output.get(i) / z.floatValue()));
            return;
        }
        float max = output.get(0);
        float z = 0.0f;
        for (int i = 0; i < osz_; i++) {
            max = FastMath.max(output.get(i), max);
        }
        for (int i = 0; i < osz_; i++) {
            output.set(i, (float) FastMath.exp(output.get(i) - max));
            z += output.get(i);
        }
        for (int i = 0; i < osz_; i++) {
            output.set(i, output.get(i) / z);
        }
    }

    private void computeOutputSoftmax() {
        computeOutputSoftmax(hidden_, output_);
    }

    /**
     * <pre>{@code real Model::softmax(int32_t target, real lr) {
     *  grad_.zero();
     *  computeOutputSoftmax();
     *  for (int32_t i = 0; i < osz_; i++) {
     *      real label = (i == target) ? 1.0 : 0.0;
     *      real alpha = lr * (label - output_[i]);
     *      grad_.addRow(*wo_, i, alpha);
     *      wo_->addRow(hidden_, i, alpha);
     *  }
     *  return -log(output_[target]);
     * }}</pre>
     *
     * @param target int32_t
     * @param lr     float
     * @return float
     */
    private float softmax(int target, float lr) {
        grad_.clear();
        computeOutputSoftmax();
        IntStream ints = IntStream.range(0, osz_);
        ints.forEach(i -> {
            float label = i == target ? 1.0f : 0.0f;
            float alpha = lr * (label - output_.get(i));
            grad_.addRow(wo_, i, alpha);
            wo_.addRow(hidden_, i, alpha);
        });
        return -log(output_.get(target));
    }

    /**
     * <pre>{@code void Model::computeHidden(const std::vector<int32_t>& input, Vector& hidden) const {
     *  assert(hidden.size() == hsz_);
     *  hidden.zero();
     *  for (auto it = input.cbegin(); it != input.cend(); ++it) {
     *      if(quant_) {
     *          hidden.addRow(*qwi_, *it);
     *      } else {
     *          hidden.addRow(*wi_, *it);
     *      }
     *  }
     *  hidden.mul(1.0 / input.size());
     * }}</pre>
     *
     * @param input  List of ints (int32_t)
     * @param hidden {@link Vector}
     */
    private void computeHidden(List<Integer> input, Vector hidden) {
        Validate.isTrue(hidden.size() == dim, "Wrong size of hidden vector: " + hidden.size() + "!=" + dim);
        hidden.clear();
        input.forEach(it -> {
            if (isQuant()) {
                hidden.addRow(qwi_, it);
            } else {
                hidden.addRow(wi_, it);
            }
        });
        hidden.mul(1.0f / input.size());
    }

    /**
     * <pre>{@code
     * void Model::predict(const std::vector<int32_t>& input, int32_t k, std::vector<std::pair<real, int32_t>>& heap, Vector& hidden, Vector& output) const {
     *  if (k <= 0) {
     *      throw std::invalid_argument("k needs to be 1 or higher!");
     *  }
     *  if (args_->model != model_name::sup) {
     *      throw std::invalid_argument("Model needs to be supervised for prediction!");
     *  }
     *  heap.reserve(k + 1);
     *  computeHidden(input, hidden);
     *  if (args_->loss == loss_name::hs) {
     *      dfs(k, 2 * osz_ - 2, 0.0, heap, hidden);
     *  } else {
     *      findKBest(k, heap, hidden, output);
     *  }
     *  std::sort_heap(heap.begin(), heap.end(), comparePairs);
     * }}</pre>
     *
     * @param input  List of ints
     * @param k      int
     * @param hidden {@link Vector}
     * @param output {@link Vector}
     * @return {@link TreeMultimap}
     */
    public TreeMultimap<Float, Integer> predict(List<Integer> input, int k, Vector hidden, Vector output) {
        if (k <= 0) {
            throw new IllegalArgumentException("k needs to be 1 or higher!");
        }
        if (!ModelName.SUP.equals(model)) {
            throw new IllegalArgumentException("Model needs to be supervised for prediction!");
        }
        TreeMultimap<Float, Integer> heap = TreeMultimap.create(HEAP_PROBABILITY_COMPARATOR, HEAP_LABEL_COMPARATOR);
        computeHidden(input, hidden);
        if (LossName.HS == loss) {
            dfs(k, 2 * osz_ - 2, 0.0f, heap, hidden);
        } else {
            findKBest(k, heap, hidden, output);
        }
        return heap;
    }

    /**
     * <pre>{@code
     * void Model::predict(const std::vector<int32_t>& input, int32_t k, std::vector<std::pair<real, int32_t>>& heap) {
     *  predict(input, k, heap, hidden_, output_);
     * }}</pre>
     *
     * @param input List of ints
     * @param k     int
     * @return {@link TreeMultimap}
     */
    public TreeMultimap<Float, Integer> predict(List<Integer> input, int k) {
        return predict(input, k, hidden_, output_);
    }

    /**
     * <pre>{@code
     * void Model::findKBest(int32_t k, std::vector<std::pair<real, int32_t>>& heap, Vector& hidden, Vector& output) const {
     *  computeOutputSoftmax(hidden, output);
     *  for (int32_t i = 0; i < osz_; i++) {
     *      if (heap.size() == k && std_log(output[i]) < heap.front().first) {
     *          continue;
     *      }
     *      heap.push_back(std::make_pair(std_log(output[i]), i));
     *      std::push_heap(heap.begin(), heap.end(), comparePairs);
     *      if (heap.size() > k) {
     *          std::pop_heap(heap.begin(), heap.end(), comparePairs);
     *          heap.pop_back();
     *      }
     *  }
     * }
     * }}</pre>
     *
     * @param k      int
     * @param heap   {@link TreeMultimap}
     * @param hidden {@link Vector}
     * @param output {@link Vector}
     */
    private void findKBest(int k, TreeMultimap<Float, Integer> heap, Vector hidden, Vector output) {
        computeOutputSoftmax(hidden, output);
        for (int i = 0; i < osz_; i++) {
            float key = stdLog(output.get(i));
            if (heap.size() == k && key < heap.asMap().firstKey()) {
                continue;
            }
            put(heap, k, key, i);
        }
    }

    /**
     * Puts key-value pair to sorted multimap with fixed size.
     * The smallest element will be deleted if the size exceeds the limit.
     *
     * @param map     {@link TreeMultimap} the multimap
     * @param maxSize int, max size
     * @param key     the key
     * @param value   the value
     * @param <K>     type of key
     * @param <V>     type of value
     */
    private <K, V> void put(TreeMultimap<K, V> map, int maxSize, K key, V value) {
        map.put(key, value);
        if (map.size() > maxSize) {
            K last = map.asMap().lastKey();
            map.get(last).pollLast();
        }
    }

    /**
     * <pre>{@code
     * void Model::dfs(int32_t k, int32_t node, real score, std::vector<std::pair<real, int32_t>>& heap, Vector& hidden) const {
     *  if (heap.size() == k && score < heap.front().first) {
     *      return;
     *  }
     *  if (tree[node].left == -1 && tree[node].right == -1) {
     *      heap.push_back(std::make_pair(score, node));
     *      std::push_heap(heap.begin(), heap.end(), comparePairs);
     *      if (heap.size() > k) {
     *          std::pop_heap(heap.begin(), heap.end(), comparePairs);
     *          heap.pop_back();
     *      }
     *      return;
     *  }
     *  real f;
     *  if (quant_ && args_->qout) {
     *      f= qwo_->dotRow(hidden, node - osz_);
     *  } else {
     *      f= wo_->dotRow(hidden, node - osz_);
     *  }
     *  f = 1. / (1 + std::exp(-f));
     *  dfs(k, tree[node].left, score + std_log(1.0 - f), heap, hidden);
     *  dfs(k, tree[node].right, score + std_log(f), heap, hidden);
     * }}</pre>
     *
     * @param k      int32_t
     * @param node   float
     * @param score  float
     * @param heap   {@link TreeMultimap}
     * @param hidden {@link Vector}
     */
    private void dfs(int k, int node, float score, TreeMultimap<Float, Integer> heap, Vector hidden) {
        if (heap.size() == k && score < heap.asMap().firstKey()) {
            return;
        }
        if (tree.get(node).left == -1 && tree.get(node).right == -1) {
            put(heap, k, score, node);
            return;
        }
        float f;
        if (isQuant() && qout) {
            f = qwo_.dotRow(hidden, node - osz_);
        } else {
            f = wo_.dotRow(hidden, node - osz_);
        }
        f = (float) (1 / (1 + FastMath.exp(-f)));
        dfs(k, tree.get(node).left, score + stdLog(1.0f - f), heap, hidden);
        dfs(k, tree.get(node).right, score + stdLog(f), heap, hidden);
    }

    /**
     * Updates the model.
     * Used while train only.
     * <pre>{@code void Model::update(const std::vector<int32_t>& input, int32_t target, real lr) {
     *  assert(target >= 0);
     *  assert(target < osz_);
     *  if (input.size() == 0) return;
     *  computeHidden(input, hidden_);
     *  if (args_->loss == loss_name::ns) {
     *      loss_ += negativeSampling(target, lr);
     *  } else if (args_->loss == loss_name::hs) {
     *      loss_ += hierarchicalSoftmax(target, lr);
     *  } else {
     *      loss_ += softmax(target, lr);
     *  }
     *  nexamples_ += 1;
     *  if (args_->model == model_name::sup) {
     *      grad_.mul(1.0 / input.size());
     *  }
     *  for (auto it = input.cbegin(); it != input.cend(); ++it) {
     *      wi_->addRow(grad_, *it, 1.0);
     *  }
     * }}</pre>
     *
     * @param input  List of ints
     * @param target int
     * @param lr     float
     */
    void update(List<Integer> input, int target, float lr) {
        Validate.isTrue(target >= 0);
        Validate.isTrue(target < osz_);
        if (input.isEmpty()) {
            return;
        }
        Events.MODEL_COMPUTE_HIDDEN.start();
        Map<Integer, Long> inputMap;
        if (FastText.USE_PARALLEL_COMPUTATION && input.size() > PARALLEL_SIZE_THRESHOLD) {
            inputMap = input.parallelStream().collect(Collectors.groupingBy(Function.identity(), Collectors.counting()));
            hidden_.clear();
            inputMap.entrySet().parallelStream().forEach(e -> hidden_.addRow(isQuant() ? qwi_ : wi_, e.getKey(), e.getValue()));
            hidden_.mul(1.0f / input.size());
        } else {
            inputMap = Collections.emptyMap();
            computeHidden(input, hidden_);
        }
        Events.MODEL_COMPUTE_HIDDEN.end();

        Events.MODEL_LOSS_CALC.start();
        if (LossName.NS == loss) {
            loss_ += negativeSampling(target, lr);
        } else if (LossName.HS == loss) {
            loss_ += hierarchicalSoftmax(target, lr);
        } else {
            loss_ += softmax(target, lr);
        }
        Events.MODEL_LOSS_CALC.end();

        nexamples_ += 1;

        Events.MODEL_GRAD_MUL.start();
        if (ModelName.SUP == model) {
            grad_.mul(1.0f / input.size());
        }
        Events.MODEL_GRAD_MUL.end();

        Events.MODEL_INPUT_ADD_ROW.start();
        if (inputMap.isEmpty()) {
            input.forEach(it -> wi_.addRow(grad_, it, 1.0f));
        } else {
            inputMap.entrySet().parallelStream().forEach(e -> wi_.addRow(grad_, e.getKey(), e.getValue()));
        }
        Events.MODEL_INPUT_ADD_ROW.end();
    }

    /**
     * <pre>{@code
     * void Model::setTargetCounts(const std::vector<int64_t>& counts) {
     *  assert(counts.size() == osz_);
     *  if (args_->loss == loss_name::ns) {
     *      initTableNegatives(counts);
     *  }
     *  if (args_->loss == loss_name::hs) {
     *      buildTree(counts);
     *  }
     * }
     * }</pre>
     *
     * @param counts List of longs (int64_t)
     */
    public void setTargetCounts(List<Long> counts) {
        Validate.isTrue(counts.size() == osz_);
//        System.out.println("loss name: " + loss);
        if (LossName.NS == loss) {
            initTableNegatives(counts);
        }
        if (LossName.HS == loss) {
            buildTree(counts);
        }
    }

    /**
     * <pre>{@code
     * void Model::initTableNegatives(const std::vector<int64_t>& counts) {
     *  real z = 0.0;
     *  for (size_t i = 0; i < counts.size(); i++) {
     *      z += pow(counts[i], 0.5);
     *  }
     *  for (size_t i = 0; i < counts.size(); i++) {
     *      real c = pow(counts[i], 0.5);
     *      for (size_t j = 0; j < c * NEGATIVE_TABLE_SIZE / z; j++) {
     *          negatives.push_back(i);
     *      }
     *  }
     *  std::shuffle(negatives.begin(), negatives.end(), rng);
     * }
     * }</pre>
     *
     * @param counts List of longs (int64_t)
     */
    private void initTableNegatives(List<Long> counts) {
        negatives = new ArrayList<>(counts.size());
        /*
        if (FastText.USE_PARALLEL_COMPUTATION && counts.size() > PARALLEL_SIZE_THRESHOLD) {
        	System.out.println("fasttext use parallel computation!");
            List<Integer> sync = Collections.synchronizedList(negatives);
            System.out.println("sunchronizedList wad synchronized");
            double z = counts.parallelStream().mapToDouble(FastMath::sqrt).sum();
            System.out.println("z: " + z);
            IntStream.range(0, counts.size()).parallel().forEach(i -> {
                double c = FastMath.sqrt(counts.get(i)) * NEGATIVE_TABLE_SIZE / z;
                for (int j = 0; j < c; j++) {
                    sync.add(i);
                }
            });
            System.out.println("finish!!!!");
        } else {
        */
            double z = 0.0;
            for (long count : counts) {
                z += FastMath.sqrt(count);
            }
            for (int i = 0; i < counts.size(); i++) {
                double c = FastMath.sqrt(counts.get(i)) * NEGATIVE_TABLE_SIZE / z;
                for (int j = 0; j < c; j++) {
                    negatives.add(i);
                }
            }
        //}
        Collections.shuffle(negatives, new RandomAdaptor(rng));
    }

    /**
     * <pre>{@code int32_t Model::getNegative(int32_t target) {
     *  int32_t negative;
     *  do {
     *      negative = negatives_[negpos];
     *      negpos = (negpos + 1) % negatives_.size();
     *  } while (target == negative);
     *  return negative;
     * }</pre>
     *
     * @param target
     * @return
     */
    private int getNegative(int target) {
        int negative;
        do {
            negative = negatives.get(negpos);
            negpos = (negpos + 1) % negatives.size();
        } while (target == negative);
        return negative;
    }

    /**
     * <pre>{@code void Model::buildTree(const std::vector<int64_t>& counts) {
     *  tree.resize(2 * osz_ - 1);
     *  for (int32_t i = 0; i < 2 * osz_ - 1; i++) {
     *      tree[i].parent = -1;
     *      tree[i].left = -1;
     *      tree[i].right = -1;
     *      tree[i].count = 1e15;
     *      tree[i].binary = false;
     *  }
     *  for (int32_t i = 0; i < osz_; i++) {
     *      tree[i].count = counts[i];
     *  }
     *  int32_t leaf = osz_ - 1;
     *  int32_t node = osz_;
     *  for (int32_t i = osz_; i < 2 * osz_ - 1; i++) {
     *      int32_t mini[2];
     *      for (int32_t j = 0; j < 2; j++) {
     *          if (leaf >= 0 && tree[leaf].count < tree[node].count) {
     *              mini[j] = leaf--;
     *          } else {
     *              mini[j] = node++;
     *          }
     *      }
     *      tree[i].left = mini[0];
     *      tree[i].right = mini[1];
     *      tree[i].count = tree[mini[0]].count + tree[mini[1]].count;
     *      tree[mini[0]].parent = i;
     *      tree[mini[1]].parent = i;
     *      tree[mini[1]].binary = true;
     *  }
     *  for (int32_t i = 0; i < osz_; i++) {
     *      std::vector<int32_t> path;
     *      std::vector<bool> code;
     *      int32_t j = i;
     *      while (tree[j].parent != -1) {
     *          path.push_back(tree[j].parent - osz_);
     *          code.push_back(tree[j].binary);
     *          j = tree[j].parent;
     *      }
     *      paths.push_back(path);
     *      codes.push_back(code);
     *  }
     * }}</pre>
     *
     * @param counts List of longs (int64_t)
     */
    private void buildTree(List<Long> counts) {
        paths = new ArrayList<>(osz_);
        codes = new ArrayList<>(osz_);
        tree = new ArrayList<>(2 * osz_ - 1);
        for (int i = 0; i < 2 * osz_ - 1; i++) {
            Node node = new Node();
            node.parent = -1;
            node.left = -1;
            node.right = -1;
            node.count = 1000_000_000_000_000L;// 1e15;
            node.binary = false;
            tree.add(i, node);
        }
        for (int i = 0; i < osz_; i++) {
            tree.get(i).count = counts.get(i);
        }
        int leaf = osz_ - 1;
        int node = osz_;
        for (int i = osz_; i < 2 * osz_ - 1; i++) {
            int[] mini = new int[2];
            for (int j = 0; j < 2; j++) {
                if (leaf >= 0 && tree.get(leaf).count < tree.get(node).count) {
                    mini[j] = leaf--;
                } else {
                    mini[j] = node++;
                }
            }
            tree.get(i).left = mini[0];
            tree.get(i).right = mini[1];
            tree.get(i).count = tree.get(mini[0]).count + tree.get(mini[1]).count;
            tree.get(mini[0]).parent = i;
            tree.get(mini[1]).parent = i;
            tree.get(mini[1]).binary = true;
        }
        for (int i = 0; i < osz_; i++) {
            List<Integer> path = new ArrayList<>();
            List<Boolean> code = new ArrayList<>();
            int j = i;
            while (tree.get(j).parent != -1) {
                path.add(tree.get(j).parent - osz_);
                code.add(tree.get(j).binary);
                j = tree.get(j).parent;
            }
            paths.add(path);
            codes.add(code);
        }
    }

    /**
     * <pre>{@code real Model::getLoss() const {
     *  return loss_ / nexamples_;
     * }</pre>
     *
     * @return float
     */
    public float getLoss() {
        return loss_ / nexamples_;
    }

    /**
     * <pre>{@code void Model::initSigmoid() {
     *  for (int i = 0; i < SIGMOID_TABLE_SIZE + 1; i++) {
     *      real x = real(i * 2 * MAX_SIGMOID) / SIGMOID_TABLE_SIZE - MAX_SIGMOID;
     *      t_sigmoid_.push_back(1.0 / (1.0 + std::exp(-x)));
     *  }
     * }}</pre>
     */
    private void initSigmoid() {
        t_sigmoid = new float[SIGMOID_TABLE_SIZE + 1];
        for (int i = 0; i < SIGMOID_TABLE_SIZE + 1; i++) {
            float x = i * 2f * MAX_SIGMOID / SIGMOID_TABLE_SIZE - MAX_SIGMOID;
            t_sigmoid[i] = (float) (1 / (1 + FastMath.exp(-x)));
        }
    }

    /**
     * <pre>{@code void Model::initLog() {
     *  for (int i = 0; i < LOG_TABLE_SIZE + 1; i++) {
     *      real x = (real(i) + 1e-5) / LOG_TABLE_SIZE;
     *      t_log_.push_back(std::log(x));
     *  }
     * }}</pre>
     */
    private void initLog() {
        t_log = new float[LOG_TABLE_SIZE + 1];
        for (int i = 0; i < LOG_TABLE_SIZE + 1; i++) {
            float x = (i + 1e-5f) / LOG_TABLE_SIZE;
            t_log[i] = (float) FastMath.log(x);
        }
    }

    /**
     * <pre>{@code real Model::log(real x) const {
     *  if (x > 1.0) {
     *      return 0.0;
     *  }
     *  int64_t i = int64_t(x * LOG_TABLE_SIZE);
     *  return t_log_[i];
     * }
     * }}</pre>
     *
     * @param x float
     * @return float
     */
    private float log(float x) {
        if (x > 1.0) {
            return 0.0f;
        }
        long i = (long) (x * LOG_TABLE_SIZE);
        return t_log[Ints.checkedCast(i)];
    }

    /**
     * <pre>{@code real Model::std_log(real x) const {
     * return std::log(x+1e-5);
     * }}</pre>
     *
     * @param x float
     * @return float
     */
    private float stdLog(float x) {
        return (float) FastMath.log(x + 1e-5);
    }

    /**
     * <pre>{@code real Model::sigmoid(real x) const {
     *  if (x < -MAX_SIGMOID) {
     *      return 0.0;
     *  } else if (x > MAX_SIGMOID) {
     *      return 1.0;
     *  } else {
     *      int64_t i = int64_t((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2);
     *      return t_sigmoid_[i];
     *  }
     * }
     * }</pre>
     *
     * @param x float
     * @return float
     */
    private float sigmoid(float x) {
        if (x < -MAX_SIGMOID) {
            return 0.0f;
        } else if (x > MAX_SIGMOID) {
            return 1.0f;
        } else {
            long i = (long) ((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2);
            return t_sigmoid[Ints.checkedCast(i)];
        }
    }

    /**
     * model.h:
     * struct Node {
     * int32_t parent;
     * int32_t left;
     * int32_t right;
     * int64_t count;
     * bool binary;
     * };
     */
    private class Node {
        int parent;
        int left;
        int right;
        long count;
        boolean binary;
    }
}
