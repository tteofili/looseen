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

import java.io.IOException;
import java.util.List;
import java.util.Map;

import org.apache.lucene.classification.ClassificationResult;
import org.apache.lucene.classification.Classifier;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.util.BytesRef;

/**
 * a completely query based classifier, each class is identified by a query, score is assigned by either looking at a
 * query results' max score or no. of hits.
 */
public class QueryingClassifier implements Classifier<BytesRef> {

    private final Map<String, Query> queriesPerClass;
    private final IndexSearcher indexSearcher;
    private boolean useCounts;

    public QueryingClassifier(Map<String, Query> queriesPerClass, IndexSearcher indexSearcher, boolean useCounts) {
        this.queriesPerClass = queriesPerClass;
        this.indexSearcher = indexSearcher;
        this.useCounts = useCounts;
    }

    public QueryingClassifier(Map<String, Query> queriesPerClass, IndexSearcher indexSearcher) {
        this(queriesPerClass, indexSearcher, false);
    }

    @Override
    public ClassificationResult<BytesRef> assignClass(String text) throws IOException {
        ClassificationResult<BytesRef> result = null;
        for (Map.Entry<String, Query> entry : queriesPerClass.entrySet()) {
            TopDocs search = indexSearcher.search(entry.getValue(), 1);
            float score;
            if (useCounts) {
                score = search.totalHits;
            } else {
                score = search.getMaxScore();
            }

            if (result == null) {
                result = new ClassificationResult<>(new BytesRef(entry.getKey()), score);
            } else if (score > result.getScore()) {
                result = new ClassificationResult<>(new BytesRef(entry.getKey()), score);
            }
        }
        return result;
    }

    @Override
    public List<ClassificationResult<BytesRef>> getClasses(String text) throws IOException {
        throw new RuntimeException("not implemented");
    }

    @Override
    public List<ClassificationResult<BytesRef>> getClasses(String text, int max) throws IOException {
        throw new RuntimeException("not implemented");
    }
}
