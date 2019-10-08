package com.shkim.fasttext.module;

import org.apache.commons.lang.Validate;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.util.FastMath;

import com.shkim.fasttext.io.FTInputStream;
import com.shkim.fasttext.io.FTOutputStream;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * The matrix.
 * see <a href='https://github.com/facebookresearch/fastText/blob/master/src/model.cc'>matrix.cc</a> and
 * <a href='https://github.com/facebookresearch/fastText/blob/master/src/model.h'>matrix.h</a>
 */
public class Matrix {

    private static final int PARALLEL_SIZE_THRESHOLD = Integer.parseInt(System.getProperty("parallel.matrix.threshold",
            String.valueOf(FastText.PARALLEL_THRESHOLD_FACTOR * 100)));

    private float[][] data;

    protected int m; // vocabSize
    protected int n; // layer1Size

    protected Matrix() {
    }

    public Matrix(int m, int n) {
        Validate.isTrue(m > 0, "Wrong m-size: " + m);
        Validate.isTrue(n > 0, "Wrong n-size: " + n);
        this.m = m;
        this.n = n;
        this.data = new float[m][n];
    }

    public Matrix copy() {
        Matrix res = new Matrix(m, n);
        for (int i = 0; i < m; i++) {
            System.arraycopy(data[i], 0, res.data[i], 0, n);
        }
        return res;
    }

    float[] flatData() {
        float[] res = new float[m * n];
        for (int i = 0; i < m; i++) {
            System.arraycopy(data[i], 0, res, i * n, n);
        }
        return res;
    }

    float[][] data() {
        return data;
    }

    /**
     * Returns matrix data as collection of vectors.
     *
     * @return List of {@link Vector}s
     */
    public List<Vector> getData() {
        return Collections.unmodifiableList(Arrays.stream(data).map(Vector::new).collect(Collectors.toList()));
    }

    public boolean isEmpty() {
        return m == 0 || n == 0;
    }

    public int getM() {
        return m;
    }

    public int getN() {
        return n;
    }

    public long size() {
        return (long) n * m;
    }

    public float get(int i, int j) {
        validateMIndex(i);
        validateNIndex(j);
        return at(i, j);
    }

    float at(int i, int j) {
        return data[i][j];
    }

    public void set(int i, int j, float value) {
        validateMIndex(i);
        validateNIndex(j);
        put(i, j, value);
    }

    void put(int i, int j, float value) {
        data[i][j] = value;
    }

    public void compute(int i, int j, DoubleUnaryOperator operator) {
        Objects.requireNonNull(operator, "Null operator");
        data[i][j] = (float) operator.applyAsDouble(data[i][j]);
    }

    void validateMIndex(int i) {
        Validate.isTrue(i >= 0 && i < m, "First index (" + i + ") is out of range [0, " + m + ")");
    }

    void validateNIndex(int j) {
        Validate.isTrue(j >= 0 && j < n, "Second index (" + j + ") is out of range [0, " + n + ")");
    }

    void validateNVector(Vector vector) {
        Validate.isTrue(Objects.requireNonNull(vector, "Null vector").size() == n, "Wrong vector size: " + vector.size() + " (!= " + n + ")");
    }

    void validateMVector(Vector vector) {
        Validate.isTrue(Objects.requireNonNull(vector, "Null vector").size() == m, "Wrong vector size: " + vector.size() + " (!= " + m + ")");
    }

    boolean isQuant() {
        return false;
    }

    /**
     * <pre>{@code
     * void Matrix::uniform(real a) {
     *  std::minstd_rand rng(1);
     *  std::uniform_real_distribution<> uniform(-a, a);
     *  for (int64_t i = 0; i < (m_ * n_); i++) {
     *      data_[i] = uniform(rng);
     *  }
     * }
     * }</pre>
     *
     * @param rnd   {@link RandomGenerator}
     * @param bound float, distribution bound
     */
    public void uniform(RandomGenerator rnd, float bound) {
        // don't use parallel optimization:
        // the order of setting random is important to have the same prediction result as for c++ version for supervised model. wtf ?
        UniformRealDistribution uniform = new UniformRealDistribution(rnd, -bound, bound);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                data[i][j] = (float) uniform.sample();
            }
        }
    }

    /**
     * <pre>{@code real Matrix::dotRow(const Vector& vec, int64_t i) const {
     *  assert(i >= 0);
     *  assert(i < m_);
     *  assert(vec.size() == n_);
     *  real d = 0.0;
     *  for (int64_t j = 0; j < n_; j++) {
     *      d += at(i, j) * vec.data_[j];
     *  }
     *  if (std::isnan(d)) {
     *      throw std::runtime_error("Encountered NaN.");
     *  }
     *  return d;
     * }}</pre>
     *
     * @param vector {@link Vector}
     * @param i      m-dimensional index
     * @return float
     */
    public float dotRow(Vector vector, int i) {
        validateMIndex(i);
        validateNVector(vector);
        float d;
        if (FastText.USE_PARALLEL_COMPUTATION && n > PARALLEL_SIZE_THRESHOLD) {
            d = (float) IntStream.range(0, n).parallel().mapToDouble(j -> data[i][j] * vector.get(j)).sum();
        } else {
            d = 0;
            for (int j = 0; j < n; j++) {
                d += data[i][j] * vector.get(j);
            }
        }
        if (Float.isNaN(d)) {
            throw new IllegalStateException("Encountered NaN.");
        }
        return d;
    }

    /**
     * <pre>{@code void Matrix::addRow(const Vector& vec, int64_t i, real a) {
     *  assert(i >= 0);
     *  assert(i < m_);
     *  assert(vec.size() == n_);
     *  for (int64_t j = 0; j < n_; j++) {
     *      data_[i * n_ + j] += a * vec.data_[j];
     *  }
     * }}</pre>
     *
     * @param vector {@link Vector}
     * @param index  m-dimensional index
     * @param factor float multiplier
     */
    public void addRow(Vector vector, int index, float factor) {
        validateMIndex(index);
        validateNVector(vector);
        if (FastText.USE_PARALLEL_COMPUTATION && n > PARALLEL_SIZE_THRESHOLD) {
            IntStream.range(0, n).parallel().forEach(j -> data[index][j] += factor * vector.get(j));
            return;
        }
        for (int j = 0; j < n; j++) {
            data[index][j] += factor * vector.get(j);
        }
    }

    /**
     * <pre>{@code void Matrix::multiplyRow(const Vector& nums, int64_t ib, int64_t ie) {
     *  if (ie == -1) {
     *      ie = m_;
     *  }
     *  assert(ie <= nums.size());
     *  for (auto i = ib; i < ie; i++) {
     *      real n = nums[i-ib];
     *      if (n != 0) {
     *          for (auto j = 0; j < n_; j++) {
     *              at(i, j) *= n;
     *          }
     *      }
     *  }
     * }}</pre>
     *
     * @param vector {@link Vector}
     * @param start  int (orig: int64_t)
     * @param end    int (orig: int64_t)
     */
    protected void multiplyRow(Vector vector, int start, int end) {
        rowOp(vector, start, end, (left, right) -> left * right);
    }

    public void multiplyRow(Vector vector) {
        multiplyRow(vector, 0, -1);
    }

    /**
     * <pre>{@code void Matrix::divideRow(const Vector& denoms, int64_t ib, int64_t ie) {
     *  if (ie == -1) {
     *      ie = m_;
     *  }
     *  assert(ie <= denoms.size());
     *  for (auto i = ib; i < ie; i++) {
     *      real n = denoms[i-ib];
     *      if (n != 0) {
     *          for (auto j = 0; j < n_; j++) {
     *              at(i, j) /= n;
     *          }
     *      }
     *  }
     * }}</pre>
     *
     * @param start, int.
     * @param end,   int. {@code -1} to use matrix m-size
     * @param vector {@link Vector}
     */
    protected void divideRow(Vector vector, int start, int end) {
        rowOp(vector, start, end, (left, right) -> left / right);
    }

    public void divideRow(Vector vector) {
        divideRow(vector, 0, -1);
    }

    protected void rowOp(Vector vector, int start, int end, DoubleBinaryOperator op) {
        if (end == -1) {
            end = m;
        }
        Validate.isTrue(end <= vector.size());
        Validate.isTrue(end >= start);
        if (FastText.USE_PARALLEL_COMPUTATION && end - start > PARALLEL_SIZE_THRESHOLD) {
            IntStream.range(start, end).parallel().forEach(i -> vectorOp(vector, i, op, start));
            return;
        }
        for (int i = start; i < end; i++) {
            vectorOp(vector, i, op, start);
        }
    }

    private void vectorOp(Vector vector, int i, DoubleBinaryOperator op, int shift) {
        float val = vector.get(i - shift);
        if (val == 0) {
            return;
        }
        if (FastText.USE_PARALLEL_COMPUTATION && n > PARALLEL_SIZE_THRESHOLD) {
            IntStream.range(0, n).parallel().forEach(j -> data[i][j] = (float) op.applyAsDouble(data[i][j], val));
            return;
        }
        for (int j = 0; j < n; j++) {
            data[i][j] = (float) op.applyAsDouble(data[i][j], val);
        }
    }

    /**
     * <pre>{@code real Matrix::l2NormRow(int64_t i) const {
     *  auto norm = 0.0;
     *  for (auto j = 0; j < n_; j++) {
     *      const real v = at(i,j);
     *      norm += v * v;
     *  }
     *  if (std::isnan(norm)) {
     *      throw std::runtime_error("Encountered NaN.");
     *  }
     *  return std::sqrt(norm);
     * }}</pre>
     *
     * @param i m-dimensional index
     * @return float
     */
    private float l2NormRow(int i) {
        float norm;
        if (FastText.USE_PARALLEL_COMPUTATION && n > PARALLEL_SIZE_THRESHOLD) {
            norm = (float) IntStream.range(0, n).parallel().mapToDouble(j -> data[i][j] * data[i][j]).sum();
        } else {
            norm = 0;
            for (int j = 0; j < n; j++) {
                float v = at(i, j);
                norm += v * v;
            }
        }
        if (Float.isNaN(norm)) {
            throw new IllegalStateException("Encountered NaN.");
        }
        return (float) FastMath.sqrt(norm);
    }

    /**
     * <pre>{@code void Matrix::l2NormRow(Vector& norms) const {
     *  assert(norms.size() == m_);
     *  for (auto i = 0; i < m_; i++) {
     *      norms[i] = l2NormRow(i);
     *  }
     * }}</pre>
     *
     * @return new {@link Vector}
     */
    public Vector l2NormRow() {
        Vector res = new Vector(m);
        IntStream ints = IntStream.range(0, m);
        if (FastText.USE_PARALLEL_COMPUTATION && m > PARALLEL_SIZE_THRESHOLD) {
            ints = ints.parallel();
        }
        ints.forEach(i -> res.set(i, l2NormRow(i)));
        return res;
    }

    /**
     * <pre>{@code
     * void Matrix::save(std::ostream& out) {
     *  out.write((char*) &m_, sizeof(int64_t));
     *  out.write((char*) &n_, sizeof(int64_t));
     *  out.write((char*) data_, m_ * n_ * sizeof(real));
     * }}</pre>
     *
     * @param out {@link FTOutputStream}
     * @throws IOException if an I/O error occurs
     */
    void save(FTOutputStream out) throws IOException {
        out.writeLong(m);
        out.writeLong(n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                out.writeFloat(data[i][j]);
            }
        }
    }

    /**
     * <pre>{@code void Matrix::load(std::istream& in) {
     *  in.read((char*) &m_, sizeof(int64_t));
     *  in.read((char*) &n_, sizeof(int64_t));
     *  delete[] data_;
     *  data_ = new real[m_ * n_];
     *  in.read((char*) data_, m_ * n_ * sizeof(real));
     * }}</pre>
     *
     * @param in {@link FTInputStream}
     * @return {@link Matrix} new instance
     * @throws IOException if an I/O error occurs
     */
    static Matrix load(FTInputStream in) throws IOException {
        Matrix res = new Matrix((int) in.readLong(), (int) in.readLong());
        for (int i = 0; i < res.m; i++) {
            for (int j = 0; j < res.n; j++) {
                res.data[i][j] = in.readFloat();
            }
        }
        return res;
    }

    /**
     * Creates an empty matrix.
     *
     * @return {@link Matrix}
     */
    static Matrix empty() {
        return new Matrix();
    }

    @Override
    public String toString() {
        return String.format("%s[m=%d, n=%d]", getClass().getSimpleName(), m, n);
    }
}
