package com.github.tteofili.looseen.dl4j;

import java.io.IOException;
import java.util.Collections;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.MultiFields;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.util.BytesRef;
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.word2vec.VocabWord;

/**
 *
 */
public class FieldValuesSequenceIterator implements SequenceIterator<VocabWord> {

    private final IndexReader reader;
    private final String field;
    private final Analyzer analyzer;
    private int currentId;
    private TermsEnum iterator;

    public FieldValuesSequenceIterator(IndexReader reader, String field, Analyzer analyzer) throws IOException {
        this.reader = reader;
        this.field = field;
        this.analyzer = analyzer;
        this.currentId = 0;
        reset();
    }


    @Override
    public boolean hasMoreSequences() {
        return currentId < reader.maxDoc();
    }

    @Override
    public Sequence<VocabWord> nextSequence() {
        if (currentId >= reader.maxDoc()) {
            return null;
        }
        try {
            Sequence<VocabWord> sequence = new Sequence<>();
            Document document = reader.document(currentId, Collections.singleton(field));
            TokenStream tokenStream = document.getField(field).tokenStream(analyzer, null);
            tokenStream.addAttribute(CharTermAttribute.class);
            tokenStream.reset();
            while (tokenStream.incrementToken()) {
                CharTermAttribute attribute = tokenStream.getAttribute(CharTermAttribute.class);
                String token = attribute.toString();
                iterator.seekExact(new BytesRef(token));
                double freq = (double) iterator.totalTermFreq() / (double) iterator.docFreq();
                VocabWord vocabWord = new VocabWord(freq, token);
                sequence.addElement(vocabWord);
            }
            tokenStream.close();
            return sequence;
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
}
