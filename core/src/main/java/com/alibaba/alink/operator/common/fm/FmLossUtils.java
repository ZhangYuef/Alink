package com.alibaba.alink.operator.common.fm;

import com.alibaba.alink.common.linalg.Vector;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

public class FmLossUtils {

    /**
     * loss function interface
     */
    public interface LossFunction extends Serializable {
        /**
         * calculate loss of sample.
         *
         * @param yTruth   true label
         * @param y        label predicted by the model
         * @return         loss
         */
        double l(double yTruth, double y);

        /**
         * calculate dldy of sample
         *
         * @param yTruth  true label
         * @param y       label predicted by the model
         * @return        loss's partial derivative of y
         */
        double dldy(double yTruth, double y);
    }


    /**
     * loss function for regression task
     */
    public static class SquareLoss implements LossFunction {
        private static final long serialVersionUID = -3903508209287601504L;
        private double maxTarget;
        private double minTarget;

        public SquareLoss(double maxTarget, double minTarget) {
            this.maxTarget = maxTarget;
            this.minTarget = minTarget;
        }

        @Override
        public double l(double yTruth, double y) {
            return (yTruth - y) * (yTruth - y);
        }

        @Override
        public double dldy(double yTruth, double y) {
            // a trick borrowed from libFM
            y = Math.min(y, maxTarget);
            y = Math.max(y, minTarget);

            return 2.0 * (y - yTruth);
        }
    }


    /**
     * loss function for binary classification task
     */
    public static class LogitLoss implements LossFunction {
        private static final long serialVersionUID = -166213844104644622L;

        @Override
        public double l(double yTruth, double y) {
            // yTruth in {0, 1}
            double logit = sigmoid(y);
            if (yTruth < 0.5) {
                return -Math.log(1. - logit);
            } else if (yTruth > 0.5) {
                return -Math.log(logit);
            } else {
                throw new RuntimeException("Invalid label: " + yTruth);
            }
        }

        @Override
        public double dldy(double yTruth, double y) {
            return sigmoid(y) - yTruth;
        }

        private double sigmoid(double x) {
            return 1.0 / (1.0 + Math.exp(-x));
        }
    }

    /**
     * Calculate metrics.
     */
    public static class metricCalc {
        // calculate MAE and MSE for regression task
        public static Tuple2<Double, Double> regression(List<Tuple3<Double, Double, Vector>> labledVectors,
                                                        double[] y) {
            double mae = 0.0;
            double mse = 0.0;
            for (int i = 0; i < y.length; i++) {
                double yDiff = y[i] - labledVectors.get(i).f1;
                mae += Math.abs(yDiff);
                mse += yDiff * yDiff;
            }

            return Tuple2.of(mae, mse);
        }

        // calculate AUC and correct number for classification task
        public static Tuple2<Double, Double> classification(List<Tuple3<Double, Double, Vector>> labledVectors,
                                              double[] y) {
            Integer[] order = new Integer[y.length];
            double correctNum = 0.0;
            for (int i = 0; i < y.length; i++) {
                order[i] = i;
                if (y[i] > 0 && labledVectors.get(i).f1 > 0.5) {
                    correctNum += 1.0;
                }
                if (y[i] < 0 && labledVectors.get(i).f1 < 0.5) {
                    correctNum += 1.0;
                }
            }
            Arrays.sort(order, new java.util.Comparator<Integer>() {
                @Override
                public int compare(Integer o1, Integer o2) {
                    return Double.compare(y[o1], y[o2]);
                }
            });

            // mSum: positive sample number
            // nSum: negative sample number
            int mSum = 0;
            int nSum = 0;
            double posRankSum = 0.;
            for (int i = 0; i < order.length; i++) {
                int sampleId = order[i];
                int rank = i + 1;
                boolean isPositiveSample = labledVectors.get(sampleId).f1 > 0.5;
                if (isPositiveSample) {
                    mSum++;
                    posRankSum += rank;
                } else {
                    nSum++;
                }
            }

            double auc;
            if (mSum != 0 && nSum != 0) {
                auc = (posRankSum - 0.5 * mSum * (mSum + 1.0)) / ((double)mSum * (double)nSum);
            } else {
                auc = 0.0;
            }

            return Tuple2.of(auc, correctNum);
        }
    }
}
