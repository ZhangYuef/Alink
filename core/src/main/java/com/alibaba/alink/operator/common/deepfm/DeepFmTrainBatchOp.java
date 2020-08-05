package com.alibaba.alink.operator.common.deepfm;

import com.alibaba.alink.common.MLEnvironment;
import com.alibaba.alink.common.linalg.DenseVector;
import com.alibaba.alink.common.linalg.Vector;
import com.alibaba.alink.common.model.ModelParamName;
import com.alibaba.alink.operator.batch.BatchOperator;
import com.alibaba.alink.operator.common.classification.ann.Topology;
import com.alibaba.alink.operator.common.optim.DeepFmOptimizer;
import com.alibaba.alink.params.recommendation.DeepFmTrainParams;

import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.ml.api.misc.param.Params;

import java.util.Random;

/**
 * DeepFM model training.
 */
public class DeepFmTrainBatchOp<T extends DeepFmTrainBatchOp<T>> extends BaseDeepFmTrainBatchOp<T> {
    private static final long serialVersionUID = -7121049631706290487L;

    private DenseVector initialWeights;
    /**
     * construct function.
     *
     * @param params parameters needed by training process.
     * @param task   Fm task: maybe "classification" or "regression".
     */
    public DeepFmTrainBatchOp(Params params, String task) {
        super(params.set(ModelParamName.TASK, task));
    }

    /**
     * construct function.
     *
     * @param task
     */
    public DeepFmTrainBatchOp(String task) {
        super(new Params().set(ModelParamName.TASK, task));
    }

    /**
     * optimize function.
     *
     * @param trainData training Data.
     * @param vecSize   vector size.
     * @param params    parameters.
     * @param dim       dimension.
     * @param topology  topology for multi-layer perception network
     * @param session   environment.
     * @return
     */
    @Override
    protected DataSet<DeepFmDataFormat> optimize(DataSet<Tuple3<Double, Double, Vector>> trainData,
                                                 DataSet<Integer> vecSize,
                                                 final Params params,
                                                 final int[] dim,
                                                 final Topology topology,
                                                 MLEnvironment session) {
        final double initStdev = params.get(DeepFmTrainParams.INIT_STDEV);
        final int[] layerSize = params.get(DeepFmTrainParams.LAYERS);
        final DenseVector initialWeights = params.get(DeepFmTrainParams.INITIAL_WEIGHTS);

        DataSet<DeepFmDataFormat> initFactors = initDeepFmModel(vecSize, dim, layerSize, initialWeights,initStdev);

        DeepFmOptimizer optimizer = new DeepFmOptimizer(trainData, topology, params);
        optimizer.setWithInitFactors(initFactors);

        return optimizer.optimize();
    }

    private DataSet<DeepFmDataFormat> initDeepFmModel(DataSet<Integer> vecSize,
                                                      int[] dim,
                                                      int[] layerSize,
                                                      DenseVector initialWeights,
                                                      double initStdev) {
        // TODO: does this need getExecutionEnvironmentFromDataSets explicitly?
        return vecSize.map(new RichMapFunction<Integer, DeepFmDataFormat>() {
            private static final long serialVersionUID = 5898149131657343503L;

            @Override
            public DeepFmDataFormat map(Integer value) throws Exception {
                DeepFmDataFormat innerModel = new DeepFmDataFormat(value, dim, layerSize, initialWeights, initStdev);

                return innerModel;
            }
        });
    }
}
