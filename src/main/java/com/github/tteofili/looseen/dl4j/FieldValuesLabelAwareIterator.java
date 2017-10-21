package com.github.tteofili.looseen.dl4j;

import java.io.IOException;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.MultiFields;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.util.BytesRef;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.documentiterator.LabelsSource;

/**
 *
 */
public class FieldValuesLabelAwareIterator implements LabelAwareIterator {

    private final IndexReader reader;
    private final String field;
    private final Analyzer analyzer;
    private int currentId;
    private TermsEnum iterator;

    public FieldValuesLabelAwareIterator(IndexReader reader, String field, Analyzer analyzer) throws IOException {
        this.reader = reader;
        this.field = field;
        this.analyzer = analyzer;
        this.currentId = 0;
        reset();
    }

    @Override
    public boolean hasNextDocument() {
        return currentId < reader.maxDoc();
    }

    @Override
    public LabelledDocument nextDocument() {
        if (currentId >= reader.maxDoc()) {
            return null;
        }
        try {
            LabelledDocument labelledDocument = new LabelledDocument();
            Document document = reader.document(currentId, Collections.singleton(field));
            labelledDocument.addLabel("doc_" + currentId);
            labelledDocument.setId("doc_" + currentId);
            labelledDocument.setContent(document.getField(field).stringValue());

//            List<VocabWord> vocabWords = new LinkedList<>();
//            TokenStream tokenStream = document.getField(field).tokenStream(analyzer, null);
//            tokenStream.addAttribute(CharTermAttribute.class);
//            tokenStream.reset();
//            while (tokenStream.incrementToken()) {
//                CharTermAttribute attribute = tokenStream.getAttribute(CharTermAttribute.class);
//                String token = attribute.toString();
//                iterator.seekExact(new BytesRef(token));
//                double freq = (double) iterator.totalTermFreq() / (double) iterator.docFreq();
//                VocabWord vocabWord = new VocabWord(freq, token);
//                vocabWords.add(vocabWord);
//            }
//            tokenStream.close();
//            labelledDocument.setReferencedContent(vocabWords);
            return labelledDocument;
        } catch (IOException e) {
            throw new RuntimeException(e);
        } finally {
            currentId++;
        }
    }

    @Override
    public void reset() {
        try {
            currentId = 0;
            this.iterator = MultiFields.getTerms(reader, field).iterator();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public LabelsSource getLabelsSource() {
        return new LabelsSource("doc_" + currentId);
    }

    @Override
    public void shutdown() {
    }

    @Override
    public boolean hasNext() {
        return hasNextDocument();
    }

    @Override
    public LabelledDocument next() {
        return nextDocument();
    }
}
