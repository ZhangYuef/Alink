package com.alibaba.alink.operator.common.utils;

import com.alibaba.alink.common.linalg.DenseVector;
import com.alibaba.alink.common.linalg.SparseVector;
import com.alibaba.alink.common.linalg.Vector;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;

public class FmOptimizerUtils {

    /**
     * Calculate the value of y with given fm model.
     *
     * @param vec           input data vector.
     * @param linearItems   1-dimension vector which size is vecSize(input feature's max dimension).
     * @param factors       2-dimension vector [vecSize][NUM_FACTOR] or [vecSize*numField][NUM_FACTOR].
     * @param bias          bias for fm's linear part.
     * @param dim           dim[0] - WITH_INTERCEPT, dim[1] - WITH_LINEAR_ITEM, dim[2] - NUM_FACTOR
     * @return Tuple2, output y given by Fm part in DeepFm and product of input and summation of vixi over i.
     */
    public static Tuple2<Double, double[]> fmCalcY(Vector vec, double[] linearItems, double[][] factors,
                                                 double bias, int[] dim) {
        int[] featureIds;
        double[] featureValues;
        if (vec instanceof SparseVector) {
            featureIds = ((SparseVector) vec).getIndices();
            featureValues = ((SparseVector) vec).getValues();
        } else {
            featureIds = new int[vec.size()];
            for (int i = 0; i < vec.size(); ++i) {
                featureIds[i] = i;
            }
            featureValues = ((DenseVector) vec).getData();
        }

        double[] vx = new double[dim[2]];
        double[] v2x2 = new double[dim[2]];

        // compute y
        double y = 0.;

        if (dim[0] > 0) {
            y += bias;
        }

        for (int i = 0; i < featureIds.length; i++) {
            int featurePos = featureIds[i];
            double x = featureValues[i];

            // the linear term
            if (dim[1] > 0) {
                y += x * linearItems[featurePos];
            }
            // the quadratic term
            for (int j = 0; j < dim[2]; j++) {
                double vixi = x * factors[featurePos][j];
                vx[j] += vixi;
                v2x2[j] += vixi * vixi;
            }
        }

        for (int i = 0; i < dim[2]; i++) {
            y += 0.5 * (vx[i] * vx[i] - v2x2[i]);
        }
        return Tuple2.of(y, vx);
    }

    /**
     * Calculate gradients and update parameters with Adagrad optimization.
     *
     * @param sigmaGii  (bias, linearItems, factors) to store gradient summations.
     * @param factors   (bias, linearItems, factors) to store parameters after update.
     * @param yVx       (y, sum of vivx)
     * @param sample    sample data for a batch.
     * @param weights   weights for each input feature.
     * @param dldy      loss's partial derivative over y.
     * @param lambda    hyper-parameter for bias, linearItems and factors.
     * @param learnRate initial learning rate.
     * @param eps       smoothing term.
     * @return Tuple3 (sigmaGii, factors, weights)
     */
    public static Tuple3<Tuple3<Double, double[], double[][]>,
            Tuple3<Double, double[], double[][]>,
            double[]> calcGradient(Tuple3<Double, double[], double[][]> sigmaGii,
                                   Tuple3<Double, double[], double[][]> factors,
                                   Tuple2<Double, double[]> yVx,
                                   Tuple3<Double, Double, Vector> sample,
                                   double[] weights,
                                   double dldy,
                                   double[] lambda,
                                   int[] dim,
                                   double learnRate,
                                   double eps) {
        if (dim[0] > 0) {
            double grad = dldy + lambda[0] * factors.f0;
            sigmaGii.f0 += grad * grad;
            factors.f0 += -learnRate * grad / (Math.sqrt(sigmaGii.f0 + eps));
        }

        Tuple2<int[], double[]> idxData = transformSampleData(sample);

        for (int i = 0; i < idxData.f0.length; i++) {
            int idx = idxData.f0[i];

            if (dim[1] > 0) {
                double grad = dldy * idxData.f1[i] + lambda[1] * factors.f1[idx];
                sigmaGii.f1[idx] += grad * grad;
                factors.f1[idx] += -grad * learnRate / (Math.sqrt(sigmaGii.f1[idx] + eps));
            }

            for (int j = 0; j < dim[2]; j++) {
                double vixi = idxData.f1[i] * factors.f2[idx][j];
                double grad = dldy * idxData.f1[i] * (yVx.f1[j] - vixi)
                        + lambda[2] * factors.f2[idx][j];
                sigmaGii.f2[idx][j] += grad * grad;
                factors.f2[idx][j] += -learnRate * grad / Math.sqrt(sigmaGii.f2[idx][j] + eps);
            }
        }

        // update weights for each input feature
        for (int i = 0; i < idxData.f0.length; i++) {
            int idx = idxData.f0[i];
            weights[idx] += sample.f0;
        }

        return Tuple3.of(sigmaGii, factors, weights);
    }

    private static Tuple2<int[], double[]> transformSampleData(Tuple3<Double, Double, Vector> sample) {
        int[] indices;
        double[] vals;
        if (sample.f2 instanceof SparseVector) {
            indices = ((SparseVector)sample.f2).getIndices();
            vals = ((SparseVector)sample.f2).getValues();
        } else {
            indices = new int[sample.f2.size()];
            for (int i = 0; i < sample.f2.size(); ++i) {
                indices[i] = i;
            }
            vals = ((DenseVector)sample.f2).getData();
        }

        return Tuple2.of(indices, vals);
    }

}
