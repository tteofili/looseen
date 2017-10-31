package com.github.tteofili.looseen.dl4j;

import org.apache.lucene.index.IndexReader;
import org.apache.lucene.search.similarities.BasicStats;
import org.apache.lucene.search.similarities.SimilarityBase;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.FloatBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 *
 */
public class NeuralNetworkSimilarity extends SimilarityBase {

  private final MultiLayerNetwork network;

  public NeuralNetworkSimilarity(MultiLayerNetwork network) {
    this.network = network;
  }

  public NeuralNetworkSimilarity() {
    double learningRate = 0.1;
    WeightInit weightInit = WeightInit.XAVIER;
    Updater updater = Updater.RMSPROP;
    int lstmLayerSize = 30;
    Activation activation = Activation.SOFTPLUS;
    int noOfHiddenLayers = 1;
    int tbpttLength = 15;

    NeuralNetConfiguration.ListBuilder builder = new NeuralNetConfiguration.Builder()
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
        .learningRate(learningRate)
        .regularization(true)
        .l2(0.01)
        .seed(12345)
        .weightInit(weightInit)
        .updater(updater)
        .list()
        .layer(0, new LSTM.Builder().nIn(8).nOut(lstmLayerSize)
            .activation(activation).build());

    for (int i = 0; i < noOfHiddenLayers; i++) {
      builder = builder.layer(i+1, new LSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
          .activation(activation).build());
    }
    builder.layer(noOfHiddenLayers, new RnnOutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS).activation(activation)
        .nIn(lstmLayerSize).nOut(1).build())
        .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
        .pretrain(true).backprop(true)
        .build();

    MultiLayerNetwork net = new MultiLayerNetwork(builder.build());
    net.init();
    this.network = net;
  }

  @Override
  protected float score(BasicStats stats, float freq, float docLen) {
    int inputSize = 8;

    float[] doubles = new float[inputSize];
    doubles[0] = stats.getAvgFieldLength();
    doubles[1] = stats.getBoost();
    doubles[2] = (float) stats.getDocFreq();
    doubles[3] = (float) stats.getNumberOfDocuments();
    doubles[4] = (float) stats.getNumberOfFieldTokens();
    doubles[5] = (float) stats.getTotalTermFreq();
    doubles[6] = freq;
    doubles[7] = docLen;

    INDArray input = Nd4j.create(new FloatBuffer(doubles), new int[] {1, inputSize});
    input.divi(input.ameanNumber());

    float v = network.feedForward(input, true).get(network.getnLayers()).maxNumber().floatValue();
    return Float.isFinite(v) ? v : 0;
  }

  @Override
  public String toString() {
    return network.toString();
  }
}
