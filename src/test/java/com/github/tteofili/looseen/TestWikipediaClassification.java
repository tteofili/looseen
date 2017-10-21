package com.github.tteofili.looseen;

/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import javax.xml.stream.XMLInputFactory;
import javax.xml.stream.XMLStreamConstants;
import javax.xml.stream.XMLStreamReader;
import javax.xml.transform.stream.StreamSource;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

import com.carrotsearch.randomizedtesting.annotations.TimeoutSuite;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.it.ItalianAnalyzer;
import org.apache.lucene.classification.BM25NBClassifier;
import org.apache.lucene.classification.CachingNaiveBayesClassifier;
import org.apache.lucene.classification.Classifier;
import org.apache.lucene.classification.KNearestFuzzyClassifier;
import org.apache.lucene.classification.KNearestNeighborClassifier;
import org.apache.lucene.classification.SimpleNaiveBayesClassifier;
import org.apache.lucene.classification.utils.ConfusionMatrixGenerator;
import org.apache.lucene.classification.utils.DatasetSplitter;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.SortedSetDocValuesField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.similarities.AfterEffectB;
import org.apache.lucene.search.similarities.AfterEffectL;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.search.similarities.BasicModelG;
import org.apache.lucene.search.similarities.BasicModelP;
import org.apache.lucene.search.similarities.ClassicSimilarity;
import org.apache.lucene.search.similarities.DFRSimilarity;
import org.apache.lucene.search.similarities.DistributionLL;
import org.apache.lucene.search.similarities.DistributionSPL;
import org.apache.lucene.search.similarities.IBSimilarity;
import org.apache.lucene.search.similarities.LMDirichletSimilarity;
import org.apache.lucene.search.similarities.LMJelinekMercerSimilarity;
import org.apache.lucene.search.similarities.LambdaDF;
import org.apache.lucene.search.similarities.LambdaTTF;
import org.apache.lucene.search.similarities.Normalization;
import org.apache.lucene.search.similarities.NormalizationH1;
import org.apache.lucene.search.similarities.NormalizationH3;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.LuceneTestCase;
import org.apache.lucene.util.TimeUnits;
import org.junit.Test;

@LuceneTestCase.SuppressSysoutChecks(bugUrl = "none")
@TimeoutSuite(millis = 365 * 24 * TimeUnits.HOUR) // hopefully ~1 year is long enough ;)
@LuceneTestCase.Monster("takes a lot!")
public final class TestWikipediaClassification extends LuceneTestCase {

    private static final String PREFIX = "/Users/teofili/data";
    private static final String INDEX = PREFIX + "/itwiki/index";
    private static final String TITLE_FIELD = "title";
    private static final Pattern pattern = Pattern.compile("\\[Categoria\\:(\\w+([\\|\\s\\']\\w*)*)\\]");
    private static final String CATEGORY_FIELD = "cat";
    private static final String TEXT_FIELD = "text";

    private static boolean index = false;
    private static boolean split = true;

    @Test
    public void testItalianWikipedia() throws Exception {

        String indexProperty = System.getProperty("index");
        if (indexProperty != null) {
            try {
                index = Boolean.valueOf(indexProperty);
            } catch (Exception e) {
                // ignore
            }
        }

        String splitProperty = System.getProperty("split");
        if (splitProperty != null) {
            try {
                split = Boolean.valueOf(splitProperty);
            } catch (Exception e) {
                // ignore
            }
        }

        Path mainIndexPath = Paths.get(INDEX + "/original");
        Directory directory = FSDirectory.open(mainIndexPath);
        Path trainPath = Paths.get(INDEX + "/train");
        Path testPath = Paths.get(INDEX + "/test");
        Path cvPath = Paths.get(INDEX + "/cv");
        FSDirectory cv = null;
        FSDirectory test = null;
        FSDirectory train = null;
        DirectoryReader testReader = null;
        if (split) {
            cv = FSDirectory.open(cvPath);
            test = FSDirectory.open(testPath);
            train = FSDirectory.open(trainPath);
        }

        if (index) {
            delete(mainIndexPath);
            if (split) {
                delete(trainPath, testPath, cvPath);
            }
        }

        IndexReader reader = null;
        try {
            Collection<String> stopWordsList = Arrays.asList("di", "a", "da", "in", "per", "tra", "fra", "il", "lo", "la", "i", "gli", "le");
            CharArraySet stopWords = new CharArraySet(stopWordsList, true);
            Analyzer analyzer = new ItalianAnalyzer(stopWords);
            if (index) {

                System.out.format("Indexing Italian Wikipedia...%n");

                long startIndex = System.currentTimeMillis();
                IndexWriter indexWriter = new IndexWriter(directory, new IndexWriterConfig(analyzer));

                importWikipedia(new File(PREFIX + "/itwiki/itwiki-20150405-pages-meta-current1.xml"), indexWriter);
                importWikipedia(new File(PREFIX + "/itwiki/itwiki-20150405-pages-meta-current2.xml"), indexWriter);
                importWikipedia(new File(PREFIX + "/itwiki/itwiki-20150405-pages-meta-current3.xml"), indexWriter);
                importWikipedia(new File(PREFIX + "/itwiki/itwiki-20150405-pages-meta-current4.xml"), indexWriter);

                long endIndex = System.currentTimeMillis();
                System.out.format("Indexed %d pages in %ds %n", indexWriter.maxDoc(), (endIndex - startIndex) / 1000);

                indexWriter.close();

            }

            if (split && !index) {
                reader = DirectoryReader.open(train);
            } else {
                reader = DirectoryReader.open(directory);
            }

            if (index && split) {
                // split the index
                System.out.format("Splitting the index...%n");

                long startSplit = System.currentTimeMillis();
                DatasetSplitter datasetSplitter = new DatasetSplitter(0.1, 0);
                for (LeafReaderContext context : reader.leaves()) {
                    datasetSplitter.split(context.reader(), train, test, cv, analyzer, false, CATEGORY_FIELD, TEXT_FIELD, CATEGORY_FIELD);
                }
                reader.close();
                reader = DirectoryReader.open(train); // using the train index from now on
                long endSplit = System.currentTimeMillis();
                System.out.format("Splitting done in %ds %n", (endSplit - startSplit) / 1000);
            }

            final long startTime = System.currentTimeMillis();

            List<Classifier<BytesRef>> classifiers = new LinkedList<>();
            classifiers.add(new KNearestNeighborClassifier(reader, new ClassicSimilarity(), analyzer, null, 1, 0, 0, CATEGORY_FIELD, TEXT_FIELD));
            classifiers.add(new KNearestNeighborClassifier(reader, new BM25Similarity(), analyzer, null, 1, 0, 0, CATEGORY_FIELD, TEXT_FIELD));
            classifiers.add(new KNearestNeighborClassifier(reader, null, analyzer, null, 1, 0, 0, CATEGORY_FIELD, TEXT_FIELD));
            classifiers.add(new KNearestNeighborClassifier(reader, new LMDirichletSimilarity(), analyzer, null, 3, 1, 1, CATEGORY_FIELD, TEXT_FIELD));
            classifiers.add(new KNearestNeighborClassifier(reader, new LMJelinekMercerSimilarity(0.3f), analyzer, null, 3, 1, 1, CATEGORY_FIELD, TEXT_FIELD));
            classifiers.add(new KNearestNeighborClassifier(reader, new ClassicSimilarity(), analyzer, null, 3, 0, 0, CATEGORY_FIELD, TEXT_FIELD));
            classifiers.add(new KNearestNeighborClassifier(reader, new ClassicSimilarity(), analyzer, null, 3, 1, 1, CATEGORY_FIELD, TEXT_FIELD));
            classifiers.add(new KNearestNeighborClassifier(reader, new DFRSimilarity(new BasicModelG(), new AfterEffectB(), new NormalizationH1()), analyzer, null, 3, 1, 1, CATEGORY_FIELD, TEXT_FIELD));
            classifiers.add(new KNearestNeighborClassifier(reader, new DFRSimilarity(new BasicModelP(), new AfterEffectL(), new NormalizationH3()), analyzer, null, 3, 1, 1, CATEGORY_FIELD, TEXT_FIELD));
            classifiers.add(new KNearestNeighborClassifier(reader, new IBSimilarity(new DistributionSPL(), new LambdaDF(), new Normalization.NoNormalization()), analyzer, null, 3, 1, 1, CATEGORY_FIELD, TEXT_FIELD));
            classifiers.add(new KNearestNeighborClassifier(reader, new IBSimilarity(new DistributionLL(), new LambdaTTF(), new NormalizationH1()), analyzer, null, 3, 1, 1, CATEGORY_FIELD, TEXT_FIELD));
            classifiers.add(new MinHashClassifier(reader, TEXT_FIELD, CATEGORY_FIELD, 5, 1, 100));
            classifiers.add(new MinHashClassifier(reader, TEXT_FIELD, CATEGORY_FIELD, 10, 1, 100));
            classifiers.add(new MinHashClassifier(reader, TEXT_FIELD, CATEGORY_FIELD, 15, 1, 100));
            classifiers.add(new MinHashClassifier(reader, TEXT_FIELD, CATEGORY_FIELD, 15, 3, 100));
            classifiers.add(new MinHashClassifier(reader, TEXT_FIELD, CATEGORY_FIELD, 15, 3, 300));
            classifiers.add(new MinHashClassifier(reader, TEXT_FIELD, CATEGORY_FIELD, 5, 3, 100));
            classifiers.add(new KNearestFuzzyClassifier(reader, new ClassicSimilarity(), analyzer, null, 3, CATEGORY_FIELD, TEXT_FIELD));
            classifiers.add(new KNearestFuzzyClassifier(reader, new ClassicSimilarity(), analyzer, null, 1, CATEGORY_FIELD, TEXT_FIELD));
            classifiers.add(new KNearestFuzzyClassifier(reader, new BM25Similarity(), analyzer, null, 3, CATEGORY_FIELD, TEXT_FIELD));
            classifiers.add(new KNearestFuzzyClassifier(reader, new BM25Similarity(), analyzer, null, 1, CATEGORY_FIELD, TEXT_FIELD));
            classifiers.add(new BM25NBClassifier(reader, analyzer, null, CATEGORY_FIELD, TEXT_FIELD));
            classifiers.add(new CachingNaiveBayesClassifier(reader, analyzer, null, CATEGORY_FIELD, TEXT_FIELD));
            classifiers.add(new SimpleNaiveBayesClassifier(reader, analyzer, null, CATEGORY_FIELD, TEXT_FIELD));

            int maxdoc;

            if (split) {
                testReader = DirectoryReader.open(test);
                maxdoc = testReader.maxDoc();
            } else {
                maxdoc = reader.maxDoc();
            }

            System.out.format("Starting evaluation on %d docs...%n", maxdoc);

            ExecutorService service = Executors.newCachedThreadPool();
            List<Future<String>> futures = new LinkedList<>();
            for (Classifier<BytesRef> classifier : classifiers) {

                final IndexReader finalReader = reader;
                final DirectoryReader finalTestReader = testReader;
                futures.add(service.submit(() -> {
                    ConfusionMatrixGenerator.ConfusionMatrix confusionMatrix;
                    if (split) {
                        confusionMatrix = ConfusionMatrixGenerator.getConfusionMatrix(finalTestReader, classifier, CATEGORY_FIELD, TEXT_FIELD, 60000 * 30);
                    } else {
                        confusionMatrix = ConfusionMatrixGenerator.getConfusionMatrix(finalReader, classifier, CATEGORY_FIELD, TEXT_FIELD, 60000 * 30);
                    }

                    final long endTime = System.currentTimeMillis();
                    final int elapse = (int) (endTime - startTime) / 1000;

                    return " * " + classifier + " \n    * accuracy = " + confusionMatrix.getAccuracy() +
                            "\n    * precision = " + confusionMatrix.getPrecision() +
                            "\n    * recall = " + confusionMatrix.getRecall() +
                            "\n    * f1-measure = " + confusionMatrix.getF1Measure() +
                            "\n    * avgClassificationTime = " + confusionMatrix.getAvgClassificationTime() +
                            "\n    * time = " + elapse + " (sec)\n ";
                }));

            }
            for (Future<String> f : futures) {
                System.out.println(f.get());
            }

            Thread.sleep(10000);
            service.shutdown();

        } finally {
            try {
                if (reader != null) {
                    reader.close();
                }
                if (directory != null) {
                    directory.close();
                }
                if (test != null) {
                    test.close();
                }
                if (train != null) {
                    train.close();
                }
                if (cv != null) {
                    cv.close();
                }
                if (testReader != null) {
                    testReader.close();
                }
            } catch (Throwable e) {
                e.printStackTrace();
            }
        }
    }

    private void delete(Path... paths) throws IOException {
        for (Path path : paths) {
            if (Files.isDirectory(path)) {
                Stream<Path> pathStream = Files.list(path);
                Iterator<Path> iterator = pathStream.iterator();
                while (iterator.hasNext()) {
                    Files.delete(iterator.next());
                }
            }
        }

    }

    private static void importWikipedia(File dump, IndexWriter indexWriter) throws Exception {
        long start = System.currentTimeMillis();
        int count = 0;
        System.out.format("Importing %s...%n", dump);

        String title = null;
        String text = null;
        Set<String> cats = new HashSet<>();

        XMLInputFactory factory = XMLInputFactory.newInstance();
        StreamSource source;
        if (dump.getName().endsWith(".xml")) {
            source = new StreamSource(dump);
        } else {
            throw new RuntimeException("can index only wikipedia XML files");
        }
        XMLStreamReader reader = factory.createXMLStreamReader(source);
        while (reader.hasNext()) {
            if (count == Integer.MAX_VALUE) {
                break;
            }
            switch (reader.next()) {
                case XMLStreamConstants.START_ELEMENT:
                    if ("title".equals(reader.getLocalName())) {
                        title = reader.getElementText();
                    } else if (TEXT_FIELD.equals(reader.getLocalName())) {
                        text = reader.getElementText();
                        Matcher matcher = pattern.matcher(text);
                        int pos = 0;
                        while (matcher.find(pos)) {
                            String group = matcher.group(1);
                            String catName = group.replaceAll("\\|\\s", "").replaceAll("\\|\\*", "");
                            Collections.addAll(cats, catName.split("\\|"));
                            pos = matcher.end();
                        }
                    }
                    break;
                case XMLStreamConstants.END_ELEMENT:
                    if ("page".equals(reader.getLocalName())) {
                        Document page = new Document();
                        if (title != null) {
                            page.add(new TextField(TITLE_FIELD, title, StoredField.Store.YES));
                        }
                        if (text != null) {
                            page.add(new TextField(TEXT_FIELD, text, StoredField.Store.YES));
                        }
                        for (String cat : cats) {
                            page.add(new StringField(CATEGORY_FIELD, cat, Field.Store.YES));
                            page.add(new SortedSetDocValuesField(CATEGORY_FIELD, new BytesRef(cat)));
                        }
                        indexWriter.addDocument(page);
                        cats.clear();
                        count++;
                        if (count % 100000 == 0) {
                            indexWriter.commit();
                            System.out.format("Committed %d pages%n", count);
                        }
                    }
                    break;
            }
        }

        indexWriter.commit();

        long millis = System.currentTimeMillis() - start;
        System.out.format(
                "Imported %d pages in %d seconds (%.2fms/page)%n",
                count, millis / 1000, (double) millis / count);
    }

}