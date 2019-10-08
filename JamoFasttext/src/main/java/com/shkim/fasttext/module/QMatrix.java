package com.shkim.fasttext.module;

import java.io.IOException;
import java.util.List;
import java.util.function.IntFunction;

import org.apache.commons.math3.random.RandomGenerator;

import com.shkim.fasttext.io.FTInputStream;
import com.shkim.fasttext.io.FTOutputStream;

/**
 * See <a href='https://github.com/facebookresearch/fastText/blob/master/src/qmatrix.cc'>qmatrix.cc</a> &
 * <a href='https://github.com/facebookresearch/fastText/blob/master/src/qmatrix.h'>qmatrix.h</a>
 * <p>
 * Created by @szuev on 24.10.2017.
 */
public class QMatrix extends Matrix {

    private boolean qnorm_;
    private int codesize_;
    //uint8_t* codes_;
    private byte[] codes_;
    //uint8_t* norm_codes_;
    private byte[] normCodes;
    private ProductQuantizer pq_;
    private ProductQuantizer npq_;

    private QMatrix() {
    }

    /**
     * <pre>{@code QMatrix::QMatrix(const Matrix& mat, int32_t dsub, bool qnorm)
     *  : qnorm_(qnorm), m_(mat.m_), n_(mat.n_),
     *  codesize_(m_ * ((n_ + dsub - 1) / dsub)) {
     *  if (codesize_ > 0) {
     *      codes_ = new uint8_t[codesize_];
     *  }
     *  pq_ = std::unique_ptr<ProductQuantizer>( new ProductQuantizer(n_, dsub));
     *  if (qnorm_) {
     *      norm_codes_ = new uint8_t[m_];
     *      npq_ = std::unique_ptr<ProductQuantizer>( new ProductQuantizer(1, 1));
     *  }
     *  quantize(mat);
     * }}</pre>
     *
     * @param matrix
     * @param randomProvider
     * @param dsub
     * @param qnorm
     */
    public QMatrix(Matrix matrix, IntFunction<RandomGenerator> randomProvider, int dsub, boolean qnorm) {
        this.qnorm_ = qnorm;
        this.m = matrix.m;
        this.n = matrix.n;
        this.codesize_ = this.m * ((this.n + dsub - 1) / dsub);
        if (codesize_ > 0) {
            codes_ = new byte[codesize_];
        }
        pq_ = new ProductQuantizer(randomProvider, this.n, dsub);
        if (qnorm_) {
            normCodes = new byte[this.m];
            npq_ = new ProductQuantizer(randomProvider, 1, 1);
        }
        quantize(matrix);
    }

    ProductQuantizer getPQ() {
        return pq_;
    }

    ProductQuantizer getNPQ() {
        return npq_;
    }

    @Override
    public List<Vector> getData() {
        throw new UnsupportedOperationException();
    }

    @Override
    public float get(int i, int j) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void set(int i, int j, float value) {
        throw new UnsupportedOperationException();
    }

    @Override
    boolean isQuant() {
        return true;
    }

    @Override
    public void uniform(RandomGenerator rnd, float bound) {
        throw new UnsupportedOperationException();
    }

    /**
     * <pre>{@code void QMatrix::quantize(const Matrix& matrix) {
     *  assert(n_ == matrix.n_);
     *  assert(m_ == matrix.m_);
     *  Matrix temp(matrix);
     *  if (qnorm_) {
     *      Vector norms(temp.m_);
     *      temp.l2NormRow(norms);
     *      temp.divideRow(norms);
     *      quantizeNorm(norms);
     *  }
     *  auto dataptr = temp.data_;
     *  pq_->train(m_, dataptr);
     *  pq_->compute_codes(dataptr, codes_, m_);
     * }}</pre>
     *
     * @param matrix
     */
    private void quantize(Matrix matrix) {
        if (qnorm_) {
            matrix = matrix.copy();
            Vector norms = matrix.l2NormRow();
            matrix.divideRow(norms);
            quantizeNorm(norms);
        }
        float[] data = matrix.flatData();
        pq_.train(getM(), data);
        pq_.computeCodes(data, codes_, getM());
    }

    /**
     * <pre>{@code void QMatrix::quantizeNorm(const Vector& norms) {
     *  assert(qnorm_);
     *  assert(norms.m_ == m_);
     *  auto dataptr = norms.data_;
     *  npq_->train(m_, dataptr);
     *  npq_->compute_codes(dataptr, norm_codes_, m_);
     * }}</pre>
     *
     * @param norms
     */
    private void quantizeNorm(Vector norms) {
        npq_.train(getM(), norms.data());
        npq_.computeCodes(norms.data(), normCodes, getM());
    }

    /**
     * <pre>{@code void QMatrix::addToVector(Vector& x, int32_t t) const {
     *  real norm = 1;
     *  if (qnorm_) {
     *      norm = npq_->get_centroids(0, norm_codes_[t])[0];
     *  }
     *  pq_->addcode(x, codes_, t, norm);
     * }}</pre>
     *
     * @param x
     * @param t
     */
    void addToVector(Vector x, int t) {
        float norm = 1;
        if (qnorm_) {
            norm = npq_.getCentroids(0, normCodes[t]).get(0);
        }
        pq_.addCode(x, codes_, t, norm);
    }

    /**
     * <pre>{@code real QMatrix::dotRow(const Vector& vec, int64_t i) const {
     *  assert(i >= 0);
     *  assert(i < m_);
     *  assert(vec.size() == n_);
     *  real norm = 1;
     *  if (qnorm_) {
     *      norm = npq_->get_centroids(0, norm_codes_[i])[0];
     *  }
     *  return pq_->mulcode(vec, codes_, i, norm);
     * }}</pre>
     *
     * @param vector {@link Vector}
     * @param i m-dimensional index
     * @return float
     */
    @Override
    public float dotRow(Vector vector, int i) {
        validateMIndex(i);
        validateNVector(vector);
        float norm = 1;
        if (qnorm_) {
            norm = npq_.getCentroids(0, normCodes[i]).get(0);
        }
        return pq_.mulCode(vector, codes_, i, norm);
    }

    @Override
    public void addRow(Vector vector, int index, float factor) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void multiplyRow(Vector vector) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void divideRow(Vector vector) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Vector l2NormRow() {
        throw new UnsupportedOperationException();
    }


    /**
     * <pre>{@code void QMatrix::save(std::ostream& out) {
     *  out.write((char*) &qnorm_, sizeof(qnorm_));
     *  out.write((char*) &m_, sizeof(m_));
     *  out.write((char*) &n_, sizeof(n_));
     *  out.write((char*) &codesize_, sizeof(codesize_));
     *  out.write((char*) codes_, codesize_ * sizeof(uint8_t));
     *  pq_->save(out);
     *  if (qnorm_) {
     *      out.write((char*) norm_codes_, m_ * sizeof(uint8_t));
     *      npq_->save(out);
     *  }
     * }}</pre>
     *
     * @param out {@link FTOutputStream}
     * @throws IOException if an I/O error occurs
     */
    @Override
    void save(FTOutputStream out) throws IOException {
        out.writeBoolean(qnorm_);
        out.writeLong(m);
        out.writeLong(n);
        out.writeInt(codesize_);
        for (byte b : codes_) {
            out.writeByte(b);
        }
        pq_.save(out);
        if (!qnorm_) return;
        for (byte b : normCodes) {
            out.writeByte(b);
        }
        npq_.save(out);
    }

    /**
     * <pre>{@code
     * void QMatrix::load(std::istream& in) {
     *  in.read((char*) &qnorm_, sizeof(qnorm_));
     *  in.read((char*) &m_, sizeof(m_));
     *  in.read((char*) &n_, sizeof(n_));
     *  in.read((char*) &codesize_, sizeof(codesize_));
     *  codes_ = new uint8_t[codesize_];
     *  in.read((char*) codes_, codesize_ * sizeof(uint8_t));
     *  pq_ = std::unique_ptr<ProductQuantizer>( new ProductQuantizer());
     *  pq_->load(in);
     *  if (qnorm_) {
     *      norm_codes_ = new uint8_t[m_];
     *      in.read((char*) norm_codes_, m_ * sizeof(uint8_t));
     *      npq_ = std::unique_ptr<ProductQuantizer>( new ProductQuantizer());
     *      npq_->load(in);
     *  }
     * }
     * }</pre>
     *
     * @param factory {@link RandomGenerator} provider
     * @param in {@link FTInputStream}
     * @return {@link QMatrix} new instance
     * @throws IOException if an I/O error occurs
     */
    static QMatrix load(IntFunction<RandomGenerator> factory, FTInputStream in) throws IOException {
        QMatrix res = new QMatrix();
        res.qnorm_ = in.readBoolean();
        res.m = (int) in.readLong();
        res.n = (int) in.readLong();
        res.codesize_ = in.readInt();
        res.codes_ = new byte[res.codesize_];
        for (int i = 0; i < res.codesize_; i++) {
            res.codes_[i] = in.readByte();
        }
        res.pq_ = ProductQuantizer.load(factory, in);
        if (res.qnorm_) {
            res.normCodes = new byte[res.m];
            for (int i = 0; i < res.m; i++) {
                res.normCodes[i] = in.readByte();
            }
            res.npq_ = ProductQuantizer.load(factory, in);
        }
        return res;
    }

    static QMatrix empty() {
        return new QMatrix();
    }
}
