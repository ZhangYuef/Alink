package com.alibaba.alink.operator.common.fm;

import java.io.Serializable;

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

}
