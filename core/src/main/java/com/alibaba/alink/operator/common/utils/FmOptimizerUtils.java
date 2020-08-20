package com.alibaba.alink.operator.common.utils;

import com.alibaba.alink.common.linalg.DenseVector;
import com.alibaba.alink.common.linalg.SparseVector;
import com.alibaba.alink.common.linalg.Vector;
import org.apache.flink.api.java.tuple.Tuple2;

public class FmOptimizerUtils {

    /**
     * calculate the value of y with given fm model.
     *
     * @param vec           input data vector.
     * @param linearItems   1-dimension vector which size is vecSize(input feature's max dimension).
     * @param factors       2-dimension vector [vecSize][NUM_FACTOR] or [vecSize*numField][NUM_FACTOR].
     * @param bias          bias for fm's linear part.
     * @param dim           dim[0] - WITH_INTERCEPT, dim[1] - WITH_LINEAR_ITEM, dim[2] - NUM_FACTOR
     * @return Tuple2, output y given by Fm part in DeepFm and product of input and factors.
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
}
