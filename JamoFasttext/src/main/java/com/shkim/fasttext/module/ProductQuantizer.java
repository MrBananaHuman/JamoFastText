package com.shkim.fasttext.module;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.function.IntFunction;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.LongStream;

import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.random.RandomAdaptor;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.util.FastMath;

import com.google.common.primitives.Bytes;
import com.google.common.primitives.Floats;
import com.google.common.primitives.Ints;
import com.shkim.fasttext.io.FTInputStream;
import com.shkim.fasttext.io.FTOutputStream;

/**
 * see <a href='https://github.com/facebookresearch/fastText/blob/master/src/productquantizer.cc'>productquantizer.cc</a> and
 * <a href='https://github.com/facebookresearch/fastText/blob/master/src/productquantizer.h'>productquantizer.h</>
 *
 * Created by @szuev on 27.10.2017.
 */
public class ProductQuantizer {

    private static final int NBITS = 8;
    private static final int KSUB = 1 << NBITS;
    private static final int MAX_POINTS_PER_CLUSTER = 256;
    private static final int MAX_POINTS = MAX_POINTS_PER_CLUSTER * KSUB;
    private static final int SEED = 1234;
    private static final int NITER = 25;
    private static final float EPS = 1e-7F;

    private int dim_;
    private int nsubq_;
    private int dsub_;
    private int lastdsub_;

    private List<Float> centroids_;

    private RandomGenerator rng;

    private ProductQuantizer(IntFunction<RandomGenerator> randomProvider) {
        this.rng = randomProvider.apply(SEED);
    }

    /**
     * <pre>{@code ProductQuantizer::ProductQuantizer(int32_t dim, int32_t dsub):
     *  dim_(dim), nsubq_(dim / dsub), dsub_(dsub), centroids_(dim * ksub_), rng(seed_) {
     *  lastdsub_ = dim_ % dsub;
     *  if (lastdsub_ == 0) {lastdsub_ = dsub_;}
     *  else {nsubq_++;}
     * }
     * }</pre>
     *
     * @param randomProvider
     * @param dim
     * @param dsub
     */
    public ProductQuantizer(IntFunction<RandomGenerator> randomProvider, int dim, int dsub) {
        this(randomProvider);
        this.dim_ = dim;
        this.nsubq_ = dim / dsub;
        this.dsub_ = dsub;
        this.centroids_ = asFloatList(new float[dim * KSUB]);
        this.lastdsub_ = dim_ % dsub;
        if (this.lastdsub_ == 0) {
            this.lastdsub_ = dsub_;
        } else {
            this.nsubq_++;
        }
    }

    /**
     * <pre>{@code real* ProductQuantizer::get_centroids(int32_t m, uint8_t i) {
     *  if (m == nsubq_ - 1) {
     *      return &centroids_[m * ksub_ * dsub_ + i * lastdsub_];
     *  }
     *  return &centroids_[(m * ksub_ + i) * dsub_];
     * }}</pre>
     *
     * @param m
     * @param b
     * @return
     */
    List<Float> getCentroids(int m, byte b) {
        int i = Byte.toUnsignedInt(b);
        int index = m == nsubq_ - 1 ? m * KSUB * dsub_ + i * lastdsub_ : (m * KSUB + i) * dsub_;
        return shiftFloats(centroids_, index);
    }

    List<Float> getCentroids() {
        return centroids_;
    }

    /**
     * <pre>{@code real distL2(const real* x, const real* y, int32_t d) {
     *  real dist = 0;
     *  for (auto i = 0; i < d; i++) {
     *      auto tmp = x[i] - y[i];
     *      dist += tmp * tmp;
     *  }
     *  return dist;
     * }}</pre>
     *
     * @param x
     * @param y
     * @param d
     * @return
     */
    private float distL2(List<Float> x, List<Float> y, int d) {
        float dist = 0;
        for (int i = 0; i < d; i++) {
            float tmp = getFloat(x, i) - getFloat(y, i);
            dist += tmp * tmp;
        }
        return dist;
    }

    /**
     * <pre>{@code
     * real ProductQuantizer::assign_centroid(const real * x, const real* c0, uint8_t* code, int32_t d) const {
     *  const real* c = c0;
     *  real dis = distL2(x, c, d);
     *  code[0] = 0;
     *  for (auto j = 1; j < ksub_; j++) {
     *      c += d;
     *      real disij = distL2(x, c, d);
     *      if (disij < dis) {
     *          code[0] = (uint8_t) j;
     *          dis = disij;
     *      }
     *  }
     *  return dis;
     * }}</pre>
     *
     * @param x
     * @param c
     * @param code
     * @param d
     * @return
     */
    private float assignCentroid(List<Float> x, List<Float> c, List<Byte> code, int d) {
        float dis = distL2(x, c, d);
        code.set(0, (byte) 0);
        for (int j = 1; j < KSUB; j++) {
            c = shiftFloats(c, d);
            float disij = distL2(x, c, d);
            if (disij < dis) {
                code.set(0, (byte) j);
                dis = disij;
            }
        }
        return dis;
    }


    /**
     * <pre>{@code
     * void ProductQuantizer::Estep(const real* x, const real* centroids, uint8_t* codes, int32_t d, int32_t n) const {
     *  for (auto i = 0; i < n; i++) {
     *      assign_centroid(x + i * d, centroids, codes + i, d);
     *  }
     * }}</pre>
     *
     * @param x
     * @param centroids
     * @param codes
     * @param d
     * @param n
     */
    private void eStep(float[] x, List<Float> centroids, List<Byte> codes, int d, int n) {
        List<Float> _x = asFloatList(x);
        for (int i = 0; i < n; i++) {
            assignCentroid(shiftFloats(_x, i * d), centroids, shiftBytes(codes, i), d);
        }
    }

    /**
     * <pre>{@code
     * void ProductQuantizer::MStep(const real* x0, real* centroids, const uint8_t* codes, int32_t d, int32_t n) {
     *  std::vector<int32_t> nelts(ksub_, 0);
     *  memset(centroids, 0, sizeof(real) * d * ksub_);
     *  const real* x = x0;
     *  for (auto i = 0; i < n; i++) {
     *      auto k = codes[i];
     *      real* c = centroids + k * d;
     *      for (auto j = 0; j < d; j++) {
     *          c[j] += x[j];
     *      }
     *      nelts[k]++;
     *      x += d;
     *  }
     *  real* c = centroids;
     *  for (auto k = 0; k < ksub_; k++) {
     *      real z = (real) nelts[k];
     *      if (z != 0) {
     *          for (auto j = 0; j < d; j++) {
     *              c[j] /= z;
     *          }
     *      }
     *      c += d;
     *  }
     *  std::uniform_real_distribution<> runiform(0,1);
     *  for (auto k = 0; k < ksub_; k++) {
     *      if (nelts[k] == 0) {
     *          int32_t m = 0;
     *          while (runiform(rng) * (n - ksub_) >= nelts[m] - 1) {
     *              m = (m + 1) % ksub_;
     *          }
     *          memcpy(centroids + k * d, centroids + m * d, sizeof(real) * d);
     *          for (auto j = 0; j < d; j++) {
     *              int32_t sign = (j % 2) * 2 - 1;
     *              centroids[k * d + j] += sign * eps_;
     *              centroids[m * d + j] -= sign * eps_;
     *          }
     *          nelts[k] = nelts[m] / 2;
     *          nelts[m] -= nelts[k];
     *      }
     *  }
     * }}</pre>
     *
     * @param x0
     * @param centroids
     * @param codes
     * @param d
     * @param n
     */
    private void mStep(float[] x0, List<Float> centroids, List<Byte> codes, int d, int n) {
        List<Integer> nelts = asIntList(new int[KSUB]);
        // `memset(centroids, 0, sizeof(real) * d * ksub_);` :
        IntStream.range(0, d * KSUB).forEach(i -> centroids.set(i, 0f));

        List<Float> x = asFloatList(x0);
        for (int i = 0; i < n; i++) {
            int k = Byte.toUnsignedInt(codes.get(i));
            List<Float> c = shiftFloats(centroids, k * d);
            for (int j = 0; j < d; j++) {
                c.set(j, c.get(j) + x.get(j));
            }
            nelts.set(k, nelts.get(k) + 1);
            x = shiftFloats(x, d);
        }
        List<Float> c = centroids;
        for (int k = 0; k < KSUB; k++) {
            float z = (float) nelts.get(k);
            if (z != 0) {
                for (int j = 0; j < d; j++) {
                    c.set(j, c.get(j) / z);
                }
            }
            c = shiftFloats(c, d);
        }

        UniformRealDistribution runiform = new UniformRealDistribution(rng, 0, 1);
        for (int k = 0; k < KSUB; k++) {
            if (nelts.get(k) != 0) continue;
            int m = 0;
            while (runiform.sample() * (n - KSUB) >= nelts.get(m) - 1) {
                m = (m + 1) % KSUB;
            }
            int kd = k * d;
            int md = m * d;
            // `memcpy(centroids + k * d, centroids + m * d, sizeof(real) * d)` :
            for (int j = 0; j < d; j++) {
                centroids.set(j + kd, centroids.get(j + md));
            }
            for (int j = 0; j < d; j++) {
                float sign = ((j % 2) * 2 - 1) * EPS;
                centroids.set(j + kd, centroids.get(j + kd) + sign);
                centroids.set(j + md, centroids.get(j + md) - sign);
            }
            nelts.set(k, nelts.get(m) / 2);
            nelts.set(m, nelts.get(m) - nelts.get(k));
        }
    }


    /**
     * <pre>{@code void ProductQuantizer::train(int32_t n, const real * x) {
     *  if (n < ksub_) {
     *      std::cerr<<"Matrix too small for quantization, must have > 256 rows"<<std::endl;
     *      exit(1);
     *  }
     *  std::vector<int32_t> perm(n, 0);
     *  std::iota(perm.begin(), perm.end(), 0);
     *  auto d = dsub_;
     *  auto np = std::min(n, max_points_);
     *  real* xslice = new real[np * dsub_];
     *  for (auto m = 0; m < nsubq_; m++) {
     *      if (m == nsubq_-1) {
     *          d = lastdsub_;
     *      }
     *      if (np != n) {
     *          std::shuffle(perm.begin(), perm.end(), rng);
     *      }
     *      for (auto j = 0; j < np; j++) {
     *          memcpy (xslice + j * d, x + perm[j] * dim_ + m * dsub_, d * sizeof(real));
     *      }
     *      kmeans(xslice, get_centroids(m, 0), np, d);
     *  }
     *  delete [] xslice;
     * }}</pre>
     *
     * @param n
     * @param data
     */
    public void train(int n, float[] data) {
        if (n < KSUB) {
            throw new IllegalArgumentException("Matrix too small for quantization, must have > 256 rows");
        }
        List<Long> perm = LongStream.iterate(0, i -> ++i).limit(n).boxed().collect(Collectors.toList());
        int d = dsub_;
        int np = FastMath.min(n, MAX_POINTS);
        float[] xslice = new float[np * dsub_];
        for (int m = 0; m < nsubq_; m++) {
            if (m == nsubq_ - 1) {
                d = lastdsub_;
            }
            if (np != n) {
                Collections.shuffle(perm, new RandomAdaptor(rng));
            }
            for (int j = 0; j < np; j++) {
                // `memcpy (xslice + j * d, x + perm[j] * dim_ + m * dsub_, d * sizeof(real))` :
                long srcPos = perm.get(j) * dim_ + m * dsub_;
                if (srcPos > Integer.MAX_VALUE) {
                    throw new ArrayStoreException("Source start index too big : " + srcPos);
                }
                int dstPos = j * d;
                try {
                    System.arraycopy(data, (int) srcPos, xslice, dstPos, d);
                } catch (ArrayStoreException | ArrayIndexOutOfBoundsException e) {
                    throw new IllegalArgumentException("Can't copy arrays: " +
                            "data.length=" + data.length + ", src-pos=" + srcPos + ", " +
                            "xslice.length=" + xslice.length + ", dst-pos=" + dstPos, e);
                }
            }
            kmeans(xslice, getCentroids(m, (byte) 0), np, d);
        }
    }

    /**
     * <pre>{@code void ProductQuantizer::kmeans(const real *x, real* c, int32_t n, int32_t d) {
     *  std::vector<int32_t> perm(n,0);
     *  std::iota(perm.begin(), perm.end(), 0);
     *  std::shuffle(perm.begin(), perm.end(), rng);
     *  for (auto i = 0; i < ksub_; i++) {
     *      memcpy (&c[i * d], x + perm[i] * d, d * sizeof(real));
     *  }
     *  uint8_t* codes = new uint8_t[n];
     *  for (auto i = 0; i < niter_; i++) {
     *      Estep(x, c, codes, d, n);
     *      MStep(x, c, codes, d, n);
     *  }
     *  delete [] codes;
     * }}</pre>
     */
    private void kmeans(float[] x, List<Float> c, int n, int d) {
        List<Integer> perm = IntStream.iterate(0, operand -> ++operand).limit(n).boxed().collect(Collectors.toList());
        Collections.shuffle(perm, new RandomAdaptor(rng));
        for (int i = 0; i < KSUB; i++) {
            // `memcpy (&c[i * d], x + perm[i] * d, d * sizeof(real))` :
            int dstPos = i * d;
            int srcPos = perm.get(i) * d;
            for (int k = 0; k < d; k++) {
                c.set(k + dstPos, x[srcPos + k]);
            }
        }
        List<Byte> codes = asByteList(new byte[n]);
        for (int i = 0; i < NITER; i++) {
            eStep(x, c, codes, d, n);
            mStep(x, c, codes, d, n);
        }
    }

    /**
     * <pre>{@code void ProductQuantizer::compute_code(const real* x, uint8_t* code) const {
     *  auto d = dsub_;
     *  for (auto m = 0; m < nsubq_; m++) {
     *      if (m == nsubq_ - 1) {
     *          d = lastdsub_;
     *      }
     *      assign_centroid(x + m * dsub_, get_centroids(m, 0), code + m, d);
     *  }
     * }}</pre>
     *
     * @param x
     * @param code
     */
    private void computeCode(List<Float> x, List<Byte> code) {
        int d = dsub_;
        for (int m = 0; m < nsubq_; m++) {
            if (m == nsubq_ - 1) {
                d = lastdsub_;
            }
            assignCentroid(shiftFloats(x, m * dsub_), getCentroids(m, (byte) 0), shiftBytes(code, m), d);
        }
    }

    /**
     * <pre>{@code void ProductQuantizer::compute_codes(const real* x, uint8_t* codes, int32_t n) const {
     *  for (auto i = 0; i < n; i++) {
     *      compute_code(x + i * dim_, codes + i * nsubq_);
     *  }
     * }}</pre>
     *
     * @param data
     * @param codes
     * @param n
     */
    void computeCodes(float[] data, byte[] codes, int n) {
        List<Float> _x = asFloatList(data);
        List<Byte> _c = asByteList(codes);
        for (int i = 0; i < n; i++) {
            computeCode(shiftFloats(_x, i * dim_), shiftBytes(_c, i * nsubq_));
        }
    }

    /**
     * <pre>{@code
     * real ProductQuantizer::mulcode(const Vector& x, const uint8_t* codes, int32_t t, real alpha) const {
     *  real res = 0.0;
     *  auto d = dsub_;
     *  const uint8_t* code = codes + nsubq_ * t;
     *  for (auto m = 0; m < nsubq_; m++) {
     *      const real* c = get_centroids(m, code[m]);
     *      if (m == nsubq_ - 1) {
     *          d = lastdsub_;
     *      }
     *      for(auto n = 0; n < d; n++) {
     *          res += x[m * dsub_ + n] * c[n];
     *      }
     *  }
     *  return res * alpha;
     * }}</pre>
     *
     * @param vector
     * @param codes
     * @param t
     * @param alpha
     * @return
     */
    float mulCode(Vector vector, byte[] codes, int t, float alpha) {
        return mulCode(vector.data(), codes, t) * alpha;
    }

    private float mulCode(float[] data, byte[] codes, int t) {
        float res = 0;
        int d = dsub_;
        List<Byte> code = shiftBytes(asByteList(codes), nsubq_ * t);
        for (int m = 0; m < nsubq_; m++) {
            List<Float> c = getCentroids(m, code.get(m));
            if (m == nsubq_ - 1) {
                d = lastdsub_;
            }
            for (int n = 0; n < d; n++) {
                res += data[m * dsub_ + n] * c.get(n);
            }
        }
        return res;
    }

    /**
     * <pre>{@code
     * void ProductQuantizer::addcode(Vector& x, const uint8_t* codes, int32_t t, real alpha) const {
     *  auto d = dsub_;
     *  const uint8_t* code = codes + nsubq_ * t;
     *  for (auto m = 0; m < nsubq_; m++) {
     *      const real* c = get_centroids(m, code[m]);
     *      if (m == nsubq_ - 1) {
     *          d = lastdsub_;
     *      }
     *      for(auto n = 0; n < d; n++) {
     *          x[m * dsub_ + n] += alpha * c[n];
     *      }
     *  }
     * }}</pre>
     *
     * @param vector
     * @param codes
     * @param t
     * @param alpha
     */
    void addCode(Vector vector, byte[] codes, int t, float alpha) {
        addCode(vector.data(), codes, t, alpha);
    }

    private void addCode(float[] data, byte[] codes, int t, float alpha) {
        int d = dsub_;
        List<Byte> code = shiftBytes(asByteList(codes), nsubq_ * t);
        for (int m = 0; m < nsubq_; m++) {
            List<Float> c = getCentroids(m, code.get(m));
            if (m == nsubq_ - 1) {
                d = lastdsub_;
            }
            for (int n = 0; n < d; n++) {
                data[m * dsub_ + n] += alpha * c.get(n);
            }
        }
    }

    /**
     * <pre>{@code void ProductQuantizer::save(std::ostream& out) {
     *  out.write((char*) &dim_, sizeof(dim_));
     *  out.write((char*) &nsubq_, sizeof(nsubq_));
     *  out.write((char*) &dsub_, sizeof(dsub_));
     *  out.write((char*) &lastdsub_, sizeof(lastdsub_));
     *  out.write((char*) centroids_.data(), centroids_.size() * sizeof(real));
     * }
     * }</pre>
     *
     * @param out
     * @throws IOException
     */
    void save(FTOutputStream out) throws IOException {
        out.writeInt(dim_);
        out.writeInt(nsubq_);
        out.writeInt(dsub_);
        out.writeInt(lastdsub_);
        for (float c : centroids_) {
            out.writeFloat(c);
        }
    }

    /**
     * <pre>{@code void ProductQuantizer::load(std::istream& in) {
     *  in.read((char*) &dim_, sizeof(dim_));
     *  in.read((char*) &nsubq_, sizeof(nsubq_));
     *  in.read((char*) &dsub_, sizeof(dsub_));
     *  in.read((char*) &lastdsub_, sizeof(lastdsub_));
     *  centroids_.resize(dim_ * ksub_);
     *  for (auto i=0; i < centroids_.size(); i++) {
     *      in.read((char*) &centroids_[i], sizeof(real));
     *  }
     * }}</pre>
     *
     * @param factory {@link RandomGenerator} provider
     * @param in {@link FTInputStream}
     * @return {@link ProductQuantizer} new instance
     * @throws IOException if an I/O error occurs
     */
    static ProductQuantizer load(IntFunction<RandomGenerator> factory, FTInputStream in) throws IOException {
        ProductQuantizer res = new ProductQuantizer(factory);
        res.dim_ = in.readInt();
        res.nsubq_ = in.readInt();
        res.dsub_ = in.readInt();
        res.lastdsub_ = in.readInt();
        res.centroids_ = asFloatList(new float[res.dim_ * KSUB]);
        for (int i = 0; i < res.centroids_.size(); i++) {
            res.centroids_.set(i, in.readFloat());
        }
        return res;
    }

    public static List<Byte> asByteList(byte... unsignedByteInts) { // uint8_t
        return Bytes.asList(unsignedByteInts);
    }

    public static List<Integer> asIntList(int... values) {
        return Ints.asList(values);
    }

    public static List<Float> asFloatList(float... values) {
        return Floats.asList(values);
    }

    private float getFloat(List<Float> array, int index) {
        if (index >= array.size()) {
            return Float.NaN;
        }
        return array.get(index);
    }

    private List<Byte> shiftBytes(List<Byte> bytes, int index) {
        return shift(bytes, index);
    }

    private List<Float> shiftFloats(List<Float> floats, int index) {
        return shift(floats, index);
    }

    public static <T> List<T> shift(List<T> array, int index) {
        if (index >= array.size()) {
            return Collections.emptyList();
        }
        return array.subList(index, array.size());
    }

}
