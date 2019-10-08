package com.shkim.fasttext.module;

import com.google.common.primitives.Floats;
import com.shkim.fasttext.io.FormatUtils;

import org.apache.commons.lang.Validate;
import org.apache.commons.math3.util.FastMath;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * See <a href='https://github.com/facebookresearch/fastText/blob/master/src/vector.cc'>vector.cc</a> &
 * <a href='https://github.com/facebookresearch/fastText/blob/master/src/vector.h'>vector.h</a>
 */
public class Vector {

    private static final int PARALLEL_SIZE_THRESHOLD = Integer.parseInt(System.getProperty("parallel.vector.threshold",
            String.valueOf(FastText.PARALLEL_THRESHOLD_FACTOR * 100)));

    private float[] data;

    public Vector(int size) {
        this(new float[size]);
    }

    Vector(float[] data) {
        this.data = data;
    }

    public int size() {
        return data.length;
    }

    public float get(int i) {
        return data[i];
    }

    public void set(int i, float value) {
        data[i] = value;
    }

    public void compute(int i, DoubleUnaryOperator operator) {
        Objects.requireNonNull(operator, "Null operator");
        data[i] = (float) operator.applyAsDouble(data[i]);
    }

    float[] data() {
        return data;
    }

    /**
     * Returns a fixed-size list backed by the vectors data.
     *
     * @return List of floats
     */
    public List<Float> getData() {
        return Floats.asList(data);
    }

    public void clear() {
        data = new float[data.length];
    }

    /**
     * <pre>{@code real Vector::norm() const {
     *  real sum = 0;
     *  for (int64_t i = 0; i < m_; i++) {
     *      sum += data_[i] * data_[i];
     *  }
     *  return std::sqrt(sum);
     * }
     * }</pre>
     *
     * @return float
     */
    public float norm() {
        double sum;
        if (FastText.USE_PARALLEL_COMPUTATION && size() > PARALLEL_SIZE_THRESHOLD) {
            sum = IntStream.range(0, size()).parallel().mapToDouble(i -> data[i] * data[i]).sum();
        } else {
            sum = 0;
            for (int i = 0; i < size(); i++) {
                sum += data[i] * data[i];
            }
        }
        return (float) FastMath.sqrt(sum);
    }

    /**
     * <pre>{@code void Vector::addVector(const Vector& source) {
     *  assert(m_ == source.m_);
     *  for (int64_t i = 0; i < m_; i++) {
     *      data_[i] += source.data_[i];
     *  }
     * }}</pre>
     *
     * @param source {@link Vector}
     */
    public void addVector(Vector source) {
        addVector(source, 1);
    }

    /**
     * Sums up the vector with another one and some multiplier.
     * <pre>{@code void Vector::addVector(const Vector& source, real s) {
     *  assert(m_ == source.m_);
     *  for (int64_t i = 0; i < m_; i++) {
     *      data_[i] += s * source.data_[i];
     *  }
     * }}</pre>
     *
     * @param source {@link Vector}
     * @param s      float (see real.h)
     */
    public void addVector(Vector source, float s) {
        Validate.isTrue(size() == Objects.requireNonNull(source, "Null source vector").size(), "Wrong size of vector: " + size() + "!=" + source.size());
        if (FastText.USE_PARALLEL_COMPUTATION && size() > PARALLEL_SIZE_THRESHOLD) {
            IntStream.range(0, size()).parallel().forEach(i -> data[i] += s * source.data[i]);
            return;
        }
        for (int i = 0; i < size(); i++) {
            data[i] += s * source.data[i];
        }
    }

    /**
     * <pre>{@code
     * void Vector::mul(real a) {
     *  for (int64_t i = 0; i < m_; i++) {
     *      data_[i] *= a;
     *  }
     * }}</pre>
     *
     * @param a float
     */
    public void mul(float a) {
        if (FastText.USE_PARALLEL_COMPUTATION && size() > PARALLEL_SIZE_THRESHOLD) {
            IntStream.range(0, size()).parallel().forEach(i -> data[i] *= a);
            return;
        }
        for (int i = 0; i < size(); i++) {
            data[i] *= a;
        }
    }

    /**
     * <pre>{@code
     * void Vector::addRow(const Matrix& A, int64_t i) {
     *  assert(i >= 0);
     *  assert(i < A.m_);
     *  assert(m_ == A.n_);
     *  for (int64_t j = 0; j < A.n_; j++) {
     *      data_[j] += A.at(i, j);
     *  }
     * }}</pre>
     *
     * @param matrix {@link Matrix}
     * @param index  (int64_t originally) matrix row num (m-dimension)
     * @see #addRow(Matrix, int, float)
     */
    public void addRow(Matrix matrix, int index) {
        Validate.isTrue(index >= 0 && index < matrix.getM(), "Incompatible index (" + index + ") and matrix m-size (" + matrix.getM() + ")");
        Validate.isTrue(size() == matrix.getN(), "Wrong matrix n-size: " + size() + " != " + matrix.getN());
        if (matrix.isQuant()) {
            addQRow((QMatrix) matrix, index);
            return;
        }
        if (FastText.USE_PARALLEL_COMPUTATION && matrix.getN() > PARALLEL_SIZE_THRESHOLD) {
            IntStream.range(0, matrix.getN()).parallel().forEach(j -> data[j] += matrix.at(index, j));
            return;
        }
        for (int j = 0; j < matrix.getN(); j++) {
            data[j] += matrix.at(index, j);
        }
    }

    /**
     * <pre>{@code
     * void Vector::addRow(const QMatrix& A, int64_t i) {
     *  assert(i >= 0);
     *  A.addToVector(*this, i);
     * }}</pre>
     *
     * @param matrix {@link QMatrix}
     * @param i      (int64_t originally) matrix row num (m-dimension)
     */
    private void addQRow(QMatrix matrix, int i) {
        Validate.isTrue(i >= 0);
        matrix.addToVector(this, i);
    }

    /**
     * <pre>{@code
     * void Vector::addRow(const Matrix& A, int64_t i, real a) {
     *  assert(i >= 0);
     *  assert(i < A.m_);
     *  assert(m_ == A.n_);
     *  for (int64_t j = 0; j < A.n_; j++) {
     *      data_[j] += a * A.at(i, j);
     *  }
     * }
     * }</pre>
     *
     * @param matrix {@link Matrix}
     * @param index  m-dimension matrix coordinate
     * @param factor float, multiplier
     * @see #addRow(Matrix, int)
     */
    public void addRow(Matrix matrix, int index, float factor) {
        Validate.isTrue(index >= 0 && index < matrix.getM(), "Incompatible index (" + index + ") and matrix m-size (" + matrix.getM() + ")");
        Validate.isTrue(size() == matrix.getN(), "Wrong matrix n-size: " + size() + " != " + matrix.getN());
        if (FastText.USE_PARALLEL_COMPUTATION && matrix.getN() > PARALLEL_SIZE_THRESHOLD) {
            IntStream.range(0, matrix.getN()).parallel().forEach(j -> data[j] += factor * matrix.at(index, j));
            return;
        }
        for (int j = 0; j < matrix.getN(); j++) {
            data[j] += factor * matrix.at(index, j);
        }
    }

    /**
     * <pre>{@code
     * void Vector::mul(const Matrix& A, const Vector& vec) {
     *  assert(A.m_ == m_);
     *  assert(A.n_ == vec.m_);
     *  for (int64_t i = 0; i < m_; i++) {
     *      data_[i] = A.dotRow(vec, i);
     *  }
     * }}</pre>
     *
     * @param matrix {@link Matrix}
     * @param vector {@link Vector}
     */
    public void mul(Matrix matrix, Vector vector) {
        Validate.isTrue(matrix.getM() == size(), "Wrong matrix m-size: " + size() + " != " + matrix.getM());
        Validate.isTrue(matrix.getN() == vector.size(), "Matrix n-size (" + matrix.getN() + ") and vector size (" + vector.size() + ")  are not equal.");
        if (FastText.USE_PARALLEL_COMPUTATION && size() > PARALLEL_SIZE_THRESHOLD) {
            IntStream.range(0, size()).parallel().forEach(i -> data[i] = matrix.dotRow(vector, i));
            return;
        }
        for (int i = 0; i < size(); i++) {
            data[i] = matrix.dotRow(vector, i);
        }
    }

    /**
     * <pre>{@code int64_t Vector::argmax() {
     *  real max = data_[0];
     *  int64_t argmax = 0;
     *  for (int64_t i = 1; i < m_; i++) {
     *      if (data_[i] > max) {
     *          max = data_[i];
     *          argmax = i;
     *  }
     * }
     * return argmax;
     * }}</pre>
     *
     * @return int (original : int64_t)
     */
    public int argmax() {
        float max = get(0);
        int argmax = 0;
        for (int i = 1; i < size(); i++) {
            if (get(i) <= max) {
                continue;
            }
            max = get(i);
            argmax = i;
        }
        return argmax;
    }

    /**
     * Returns a string representation of the vector.
     * Used while printing to console and save vectors to file.
     *
     * @return vector as String
     * @see FormatUtils#toString(float)
     */
    @Override
    public String toString() {
        return getData().stream().map(FormatUtils::toString).collect(Collectors.joining(" "));
    }

    /**
     * Added to debug purposes
     *
     * @param o vector
     * @return boolean
     */
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Vector)) return false;
        Vector vector = (Vector) o;
        return Arrays.equals(data, vector.data);
    }

    /**
     * Added to debug purposes
     *
     * @return int
     */
    @Override
    public int hashCode() {
        return Arrays.hashCode(data);
    }
}
