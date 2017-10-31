package com.github.tteofili.looseen.dl4j;

//import org.apache.lucene.search.similarities.BaseSimilarityTestCase;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.RandomIndexWriter;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.LuceneTestCase;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 *
 */
//public class NeuralNetworkSimilarityTest extends BaseSimilarityTestCase {
public class NeuralNetworkSimilarityTest extends LuceneTestCase {

  @Test
  public void testScore() throws Exception {

    NeuralNetworkSimilarity sim = new NeuralNetworkSimilarity();
    Directory dir = newDirectory();
    RandomIndexWriter w = new RandomIndexWriter(random(), dir,
        newIndexWriterConfig());
    Document doc = new Document();
    doc.add(new StringField("foo", "bar", Field.Store.NO));
    doc.add(new StringField("foo", "baz", Field.Store.NO));
    w.addDocument(doc);
    doc = new Document();
    doc.add(new StringField("foo", "bar", Field.Store.NO));
    doc.add(new StringField("foo", "bar", Field.Store.NO));
    w.addDocument(doc);

    DirectoryReader reader = w.getReader();
    w.close();
    IndexSearcher searcher = newSearcher(reader);
    searcher.setSimilarity(sim);
    TopDocs topDocs = searcher.search(new TermQuery(new Term("foo", "bar")), 2);
    assertEquals(2, topDocs.totalHits);
    assertEquals(1f, topDocs.scoreDocs[0].score, 1f);
    assertEquals(1f, topDocs.scoreDocs[1].score, 1f);

    topDocs = searcher.search(new TermQuery(new Term("foo", "baz")), 1);
    assertEquals(1, topDocs.totalHits);
    assertEquals(1f, topDocs.scoreDocs[0].score, 1f);

    topDocs = searcher.search(new BoostQuery(new TermQuery(new Term("foo", "baz")), 3f), 1);
    assertEquals(1, topDocs.totalHits);
    assertEquals(1f, topDocs.scoreDocs[0].score, 1f);

    reader.close();
    dir.close();
  }

  private MultiLayerNetwork setupLSTM(double learningRate, WeightInit weightInit, Updater updater, int lstmLayerSize, Activation activation, int noOfHiddenLayers, int tbpttLength) {
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

    for (int i = 1; i < noOfHiddenLayers; i++) {
      builder = builder.layer(i, new LSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
          .activation(activation).build());
    }
    builder.layer(noOfHiddenLayers, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX)
        .nIn(lstmLayerSize).nOut(1).build())
        .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
        .pretrain(false).backprop(true)
        .build();

    MultiLayerNetwork net = new MultiLayerNetwork(builder.build());
    net.init();
    return net;
  }

}