# Looseen

playing with Lucene stuff

## Evaluation of Lucene Classifiers on 20 Newsgroups dataset


-------------------------------------------------------
 T E S T S
-------------------------------------------------------

 * KNearestNeighborClassifier{textFieldNames=[body], classFieldName='category', k=1, query=null, similarity=ClassicSimilarity} 
    * accuracy = 0.49823819591261453
    * precision = 0.662750022906968
    * recall = 0.644
    * f1-measure = 0.6532404932392698
    * avgClassificationTime = 83.0005
    * time = 672 (sec)
 
 * KNearestNeighborClassifier{textFieldNames=[body], classFieldName='category', k=1, query=null, similarity=BM25(k1=1.2,b=0.75)} 
    * accuracy = 0.5537190082644629
    * precision = 0.7096401490658643
    * recall = 0.6895
    * f1-measure = 0.6994251192171027
    * avgClassificationTime = 82.4765
    * time = 671 (sec)
 
 * KNearestNeighborClassifier{textFieldNames=[body], classFieldName='category', k=3, query=null, similarity=ClassicSimilarity} 
    * accuracy = 0.524759529747061
    * precision = 0.6830379279175388
    * recall = 0.6664999999999999
    * f1-measure = 0.6746676318457002
    * avgClassificationTime = 83.786
    * time = 674 (sec)
 
 * KNearestNeighborClassifier{textFieldNames=[body], classFieldName='category', k=3, query=null, similarity=F1EXP} 
    * accuracy = 0.543001079525009
    * precision = 0.6980933187739183
    * recall = 0.6825
    * f1-measure = 0.6902085988454955
    * avgClassificationTime = 114.8575
    * time = 736 (sec)
 
 * KNearestNeighborClassifier{textFieldNames=[body], classFieldName='category', k=3, query=null, similarity=F1LOG} 
    * accuracy = 0.5445119771183411
    * precision = 0.6972644962686834
    * recall = 0.6815
    * f1-measure = 0.6892921242069858
    * avgClassificationTime = 103.207
    * time = 713 (sec)
 
 * KNearestNeighborClassifier{textFieldNames=[body], classFieldName='category', k=3, query=null, similarity=LM Dirichlet(2000.000000)} 
    * accuracy = 0.6876190476190476
    * precision = 0.8007191293310608
    * recall = 0.7950000000000002
    * f1-measure = 0.7978493158567947
    * avgClassificationTime = 86.711
    * time = 680 (sec)
 
 * KNearestNeighborClassifier{textFieldNames=[body], classFieldName='category', k=3, query=null, similarity=LM Jelinek-Mercer(0.300000)} 
    * accuracy = 0.7059724349157733
    * precision = 0.8102908129749086
    * recall = 0.8079999999999998
    * f1-measure = 0.8091437850779882
    * avgClassificationTime = 83.4865
    * time = 673 (sec)
 
 * KNearestNeighborClassifier{textFieldNames=[body], classFieldName='category', k=3, query=null, similarity=BM25(k1=1.2,b=0.75)} 
    * accuracy = 0.6946564885496184
    * precision = 0.8022235906129224
    * recall = 0.7999999999999999
    * f1-measure = 0.8011102523397857
    * avgClassificationTime = 79.162
    * time = 665 (sec)
 
 * KNearestNeighborClassifier{textFieldNames=[body], classFieldName='category', k=3, query=null, similarity=DFR GB1} 
    * accuracy = 0.6926605504587156
    * precision = 0.8020972688350249
    * recall = 0.799
    * f1-measure = 0.8005456386362994
    * avgClassificationTime = 87.425
    * time = 681 (sec)
 
 * KNearestNeighborClassifier{textFieldNames=[body], classFieldName='category', k=3, query=null, similarity=DFR PL3(800.0)} 
    * accuracy = 0.6879053796260969
    * precision = 0.7988319471064455
    * recall = 0.7955000000000001
    * f1-measure = 0.7971624918844459
    * avgClassificationTime = 87.9455
    * time = 682 (sec)
 
 * KNearestNeighborClassifier{textFieldNames=[body], classFieldName='category', k=3, query=null, similarity=IB SPL-D} 
    * accuracy = 0.5649097983728334
    * precision = 0.7459631028746745
    * recall = 0.6924999999999998
    * f1-measure = 0.7182380246088507
    * avgClassificationTime = 92.68
    * time = 692 (sec)
 
 * KNearestNeighborClassifier{textFieldNames=[body], classFieldName='category', k=3, query=null, similarity=IB LL-L1} 
    * accuracy = 0.6983154670750383
    * precision = 0.8041118188789698
    * recall = 0.8029999999999999
    * f1-measure = 0.8035555248547892
    * avgClassificationTime = 84.4735
    * time = 675 (sec)
 
 * MinHashClassifier{min=15, hashCount=1, hashSize=100} 
    * accuracy = 0.7510729613733905
    * precision = 0.8201365115589084
    * recall = 0.8191112264118834
    * f1-measure = 0.8196235483475789
    * avgClassificationTime = 649.5380434782609
    * time = 1104 (sec)
 
 * MinHashClassifier{min=30, hashCount=3, hashSize=300} 
    * accuracy = 0.7037815126050421
    * precision = 0.7373998387051326
    * recall = 0.7263354112338665
    * f1-measure = 0.731825806766627
    * avgClassificationTime = 1906.107438016529
    * time = 1429 (sec)
 
 * MinHashClassifier{min=10, hashCount=1, hashSize=100} 
    * accuracy = 0.7924421883812747
    * precision = 0.8684939211550826
    * recall = 0.8619538215695274
    * f1-measure = 0.8652115124503503
    * avgClassificationTime = 466.95089633671085
    * time = 1106 (sec)
 
 * KNearestFuzzyClassifier{textFieldNames=[body], classFieldName='category', k=1, query=null, similarity=LM Jelinek-Mercer(0.300000)} 
    * accuracy = 0.5134205092911218
    * precision = 0.7273652032322021
    * recall = 0.6451225520511237
    * f1-measure = 0.6837797923894429
    * avgClassificationTime = 350.6059236947791
    * time = 1205 (sec)
 
 * KNearestFuzzyClassifier{textFieldNames=[body], classFieldName='category', k=1, query=null, similarity=IB LL-L1} 
    * accuracy = 0.5226565202352127
    * precision = 0.7296297239865904
    * recall = 0.6536631622345909
    * f1-measure = 0.6895605079619667
    * avgClassificationTime = 351.06325301204816
    * time = 1206 (sec)
 
 * KNearestFuzzyClassifier{textFieldNames=[body], classFieldName='category', k=1, query=null, similarity=ClassicSimilarity} 
    * accuracy = 0.6880244088482075
    * precision = 0.7985278221627009
    * recall = 0.7947985982271697
    * f1-measure = 0.7966588460197784
    * avgClassificationTime = 347.41465863453817
    * time = 1199 (sec)
 
 * KNearestFuzzyClassifier{textFieldNames=[body], classFieldName='category', k=3, query=null, similarity=ClassicSimilarity} 
    * accuracy = 0.7026194144838213
    * precision = 0.8087338831496484
    * recall = 0.8064157905586476
    * f1-measure = 0.8075731733695348
    * avgClassificationTime = 347.7931726907631
    * time = 1199 (sec)
 
 * KNearestFuzzyClassifier{textFieldNames=[body], classFieldName='category', k=1, query=null, similarity=BM25(k1=1.2,b=0.75)} 
    * accuracy = 0.6941896024464832
    * precision = 0.804097362086039
    * recall = 0.79921706864564
    * f1-measure = 0.8016497878570465
    * avgClassificationTime = 345.5245983935743
    * time = 1195 (sec)
 
 * KNearestFuzzyClassifier{textFieldNames=[body], classFieldName='category', k=3, query=null, similarity=BM25(k1=1.2,b=0.75)} 
    * accuracy = 0.7047289504036909
    * precision = 0.8101161536760554
    * recall = 0.8072529375386519
    * f1-measure = 0.8086820112425342
    * avgClassificationTime = 345.5988955823293
    * time = 1195 (sec)
 
 * KNearestFuzzyClassifier{textFieldNames=[body], classFieldName='category', k=3, query=null, similarity=F1EXP} 
    * accuracy = 0.599409811877536
    * precision = 0.738151080765962
    * recall = 0.7271584209441352
    * f1-measure = 0.7326135177333711
    * avgClassificationTime = 362.61546184738955
    * time = 1229 (sec)
 
 * KNearestFuzzyClassifier{textFieldNames=[body], classFieldName='category', k=3, query=null, similarity=F1LOG} 
    * accuracy = 0.5992619926199262
    * precision = 0.7380004783563235
    * recall = 0.7269715522572665
    * f1-measure = 0.7324445001077481
    * avgClassificationTime = 358.4615770969362
    * time = 1220 (sec)
 
 * org.apache.lucene.classification.BM25NBClassifier@59f9aa9a 
    * accuracy = 0.7333994053518335
    * precision = 0.8101928871306082
    * recall = 0.784488316199011
    * f1-measure = 0.7971334364441379
    * avgClassificationTime = 1150.7488817891374
    * time = 2306 (sec)
 
 * org.apache.lucene.classification.CachingNaiveBayesClassifier@25f05f09 
    * accuracy = 0.6043350477590007
    * precision = 0.7463507424399681
    * recall = 0.7063124999999999
    * f1-measure = 0.725779855053247
    * avgClassificationTime = 94.78507014028057
    * time = 696 (sec)
 
 * org.apache.lucene.classification.SimpleNaiveBayesClassifier@43b530e0 
    * accuracy = 0.6053711301753077
    * precision = 0.7475009863580355
    * recall = 0.706873182653093
    * f1-measure = 0.7266196175946903
    * avgClassificationTime = 489.47121752419764
    * time = 1465 (sec)
