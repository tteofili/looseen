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
package com.github.tteofili.looseen;

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.core.WhitespaceTokenizerFactory;
import org.apache.lucene.analysis.custom.CustomAnalyzer;
import org.apache.lucene.analysis.en.EnglishMinimalStemFilter;
import org.apache.lucene.analysis.minhash.MinHashFilter;
import org.apache.lucene.analysis.minhash.MinHashFilterFactory;
import org.apache.lucene.analysis.shingle.ShingleFilterFactory;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.classification.ClassificationResult;
import org.apache.lucene.classification.Classifier;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.IndexableField;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.ConstantScoreQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.RAMDirectory;
import org.apache.lucene.util.BytesRef;

/**
 * a {@link Classifier} based on LSH via queries on in memory sidecar index using {@link MinHashFilter} to index passed
 * reader's docs.
 */
public class MinHashClassifier implements Classifier<BytesRef>, Closeable {

    private static final String TEXT_FIELD = "text";
    private static final String CLASS_FIELD = "class";
    private final RAMDirectory directory;
    private final int min;
    private final int hashCount;
    private final int hashSize;

    public MinHashClassifier(IndexReader reader, String textField, String categoryField, int min, int hashCount,
                             int hashSize) {
        this.min = min;
        this.hashCount = hashCount;
        this.hashSize = hashSize;
        try {
            Analyzer analyzer = createMinHashAnalyzer(min, hashCount, hashSize);
            IndexWriterConfig config = new IndexWriterConfig(analyzer);
            directory = new RAMDirectory();
            IndexWriter writer = new IndexWriter(directory, config);
            for (int i = 0; i < reader.maxDoc(); i++) {
                Document document = new Document();
                Document d = reader.document(i);
                String textValue = d.getField(textField).stringValue();
                String categoryValue = d.getField(categoryField).stringValue();
                document.add(new TextField(TEXT_FIELD, textValue, Field.Store.NO));
                document.add(new StringField(CLASS_FIELD, categoryValue, Field.Store.YES));
                writer.addDocument(document);
            }
            writer.commit();
            writer.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        BooleanQuery.setMaxClauseCount(Integer.MAX_VALUE);
    }

    List<ClassificationResult<BytesRef>> buildListFromTopDocs(IndexSearcher searcher, String categoryFieldName, TopDocs topDocs, int k) throws IOException {
        Map<BytesRef, Integer> classCounts = new HashMap<>();
        Map<BytesRef, Double> classBoosts = new HashMap<>(); // this is a boost based on class ranking positions in topDocs
        float maxScore = topDocs.getMaxScore();
        for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
            IndexableField storableField = searcher.doc(scoreDoc.doc).getField(categoryFieldName);
            if (storableField != null) {
                BytesRef cl = new BytesRef(storableField.stringValue());
                //update count
                Integer count = classCounts.get(cl);
                if (count != null) {
                    classCounts.put(cl, count + 1);
                } else {
                    classCounts.put(cl, 1);
                }
                //update boost, the boost is based on the best score
                Double totalBoost = classBoosts.get(cl);
                double singleBoost = scoreDoc.score / maxScore;
                if (totalBoost != null) {
                    classBoosts.put(cl, totalBoost + singleBoost);
                } else {
                    classBoosts.put(cl, singleBoost);
                }
            }
        }
        List<ClassificationResult<BytesRef>> returnList = new ArrayList<>();
        List<ClassificationResult<BytesRef>> temporaryList = new ArrayList<>();
        int sumdoc = 0;
        for (Map.Entry<BytesRef, Integer> entry : classCounts.entrySet()) {
            Integer count = entry.getValue();
            Double normBoost = classBoosts.get(entry.getKey()) / count; //the boost is normalized to be 0<b<1
            temporaryList.add(new ClassificationResult<>(entry.getKey().clone(), (count * normBoost) / (double) k));
            sumdoc += count;
        }

        //correction
        if (sumdoc < k) {
            for (ClassificationResult<BytesRef> cr : temporaryList) {
                returnList.add(new ClassificationResult<>(cr.getAssignedClass(), cr.getScore() * k / (double) sumdoc));
            }
        } else {
            returnList = temporaryList;
        }
        return returnList;
    }

    @Override
    public ClassificationResult<BytesRef> assignClass(String text) throws IOException {
        DirectoryReader reader = DirectoryReader.open(directory);
        IndexSearcher searcher = new IndexSearcher(reader);
        try {
            int k = 3;
            TopDocs topDocs = searcher.search(buildQuery(TEXT_FIELD, text, min, hashCount, hashSize), k);
            if (topDocs.totalHits > 0) {
                return buildListFromTopDocs(searcher, CLASS_FIELD, topDocs, k).get(0);
//                Document document = reader.document(topDocs.scoreDocs[0].doc);
//                String category = document.getField(CLASS_FIELD).stringValue();
//                return new ClassificationResult<>(new BytesRef(category), topDocs.getMaxScore());
            } else {
                return null;
            }
        } finally {
            reader.close();
        }
    }

    @Override
    public List<ClassificationResult<BytesRef>> getClasses(String text) throws IOException {
        return null;
    }

    @Override
    public List<ClassificationResult<BytesRef>> getClasses(String text, int max) throws IOException {
        return null;
    }

    public static Analyzer createMinHashAnalyzer(int min, int hashCount, int hashSetSize) throws IOException {
        Map<String, String> sffargs = new HashMap<>();
        sffargs.put("minShingleSize", "" + min);
        sffargs.put("maxShingleSize", "" + min);
        sffargs.put("outputUnigrams", "false");
        sffargs.put("outputUnigramsIfNoShingles", "false");
        sffargs.put("tokenSeparator", " ");
        HashMap<String, String> lshffargs = new HashMap<>();
        lshffargs.put("hashCount", "" + hashCount);
        lshffargs.put("hashSetSize", "" + hashSetSize);
        CustomAnalyzer.Builder builder = CustomAnalyzer.builder()
                .withTokenizer(WhitespaceTokenizerFactory.class)
                .addTokenFilter(ShingleFilterFactory.class, sffargs)
                .addTokenFilter(MinHashFilterFactory.class, lshffargs);

        return builder.build();
    }

    private Query buildQuery(String field, String query, int min, int hashCount, int hashSetSize) throws IOException {
        Analyzer chain = createMinHashAnalyzer(min, hashCount, hashSetSize);
        ArrayList<String> tokens = getTokens(chain, field, query);
        chain.close();
        BooleanQuery.Builder builder = new BooleanQuery.Builder();
        for (String token : tokens) {
            builder.add(new ConstantScoreQuery(new TermQuery(new Term("text", token))), BooleanClause.Occur.SHOULD);
        }
        return builder.build();
    }

    private ArrayList<String> getTokens(Analyzer analyzer, String field, String value) throws IOException {
        ArrayList<String> tokens = new ArrayList<String>();
        TokenStream ts = analyzer.tokenStream(field, value);
        ts.reset();
        while (ts.incrementToken()) {
            CharTermAttribute termAttribute = ts.getAttribute(CharTermAttribute.class);
            String token = new String(termAttribute.buffer(), 0, termAttribute.length());
            tokens.add(token);
        }
        ts.end();
        ts.close();
        return tokens;
    }

    @Override
    public void close() {
        directory.close();
    }

    @Override
    public String toString() {
        return "MinHashClassifier{" +
                "min=" + min +
                ", hashCount=" + hashCount +
                ", hashSize=" + hashSize +
                '}';
    }
}
