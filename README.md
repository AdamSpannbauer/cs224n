# Stanford CS224n: NLP with Deep Learning

Links (Winter 2019 materials & lectures)

- [Schedule](http://web.stanford.edu/class/cs224n/index.html#schedule) (assignments etc)
- [Lecture Playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z)

<table class="table">
  <colgroup>
    <col style="width:20%">
    <col style="width:40%">
    <col style="width:10%">
    <col style="width:10%">
  </colgroup>
  <thead>
  <tr class="active">
    <th>Description</th>
    <th>Course Materials</th>
    <th>Events</th>
    <th>Deadlines</th>
  </tr>
  </thead>
  <tbody>
  <tr>
    <td>Word Vectors
      <br>
      [<a href="slides/cs224n-2021-lecture01-wordvecs1.pdf">slides</a>]
      <!--[<a href="https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=b2acdfb7-8038-49fb-941e-ab25012bd9ec">video</a>]-->
      [<a href="readings/cs224n-2019-notes01-wordvecs1.pdf">notes</a>]
      <br><br>
      Gensim word vectors example:
      <br>
      [<a href="materials/Gensim.zip">code</a>]
      [<a href="materials/Gensim%20word%20vector%20visualization.html">preview</a>]
    </td>
    <td>
      Suggested Readings:
      <ol>
        <li><a href="http://arxiv.org/pdf/1301.3781.pdf">Efficient Estimation of Word Representations in Vector Space</a> (original word2vec paper)</li>
        <li><a href="http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf">Distributed Representations of Words and Phrases and their Compositionality</a> (negative sampling paper)</li>
      </ol>
    </td>
    <td>
      Assignment 1 <b><font color="green">out</font></b>
      <br>
      [<a href="assignments/a1.zip">code</a>]
      <br>
      [<a href="assignments/a1_preview/exploring_word_vectors.html">preview</a>]
    </td>
    <td></td>
  </tr>

  <tr>
    <td>Word Vectors 2 and Word Window Classification
      <br>
      [<a href="slides/cs224n-2021-lecture02-wordvecs2.pdf">slides</a>]
      <!--[<a href="">video</a>]-->
      [<a href="readings/cs224n-2019-notes02-wordvecs2.pdf">notes</a>]
    </td>
    <td>
      Suggested Readings:
      <ol>
        <li><a href="http://nlp.stanford.edu/pubs/glove.pdf">GloVe: Global Vectors for Word Representation</a> (original GloVe paper)</li>
        <li><a href="http://www.aclweb.org/anthology/Q15-1016">Improving Distributional Similarity with Lessons Learned from Word Embeddings</a></li>
        <li><a href="http://www.aclweb.org/anthology/D15-1036">Evaluation methods for unsupervised word embeddings</a></li>
      </ol>
      Additional Readings:
      <ol>
        <li><a href="http://aclweb.org/anthology/Q16-1028">A Latent Variable Model Approach to PMI-based Word Embeddings</a></li>
        <li><a href="https://transacl.org/ojs/index.php/tacl/article/viewFile/1346/320">Linear Algebraic Structure of Word Senses, with Applications to Polysemy</a></li>
        <li><a href="https://papers.nips.cc/paper/7368-on-the-dimensionality-of-word-embedding.pdf">On the Dimensionality of Word Embedding</a></li>
      </ol>
    </td>
    <td></td>
    <td></td>
  </tr>

  <tr class="warning">
    <td>Python Review Session
      <br>
      [<a href="readings/cs224n-python-review-code-updated.zip">code</a>]
      [<a href="readings/cs224n-python-review-code-updated.pdf">preview</a>]
    </td>
    <td>
      <i class="fa fa-clock-o"></i> 10:00am - 11:20am<!--<br>160-124 [<a href="https://campus-map.stanford.edu/">map</a>]-->
    </td>
    <td></td>
    <td></td>
  </tr>

  <tr>
    <td>Backprop and Neural Networks
      <br>
      [<a href="slides/cs224n-2021-lecture03-neuralnets.pdf">slides</a>]
      [<a href="readings/cs224n-2019-notes03-neuralnets.pdf">notes</a>]
    </td>
    <td>
      Suggested Readings:
      <ol>
        <li><a href="readings/gradient-notes.pdf">matrix calculus notes</a></li>
        <li><a href="readings/review-differential-calculus.pdf">Review of differential calculus</a></li>
        <li><a href="http://cs231n.github.io/neural-networks-1/">CS231n notes on network architectures</a></li>
        <li><a href="http://cs231n.github.io/optimization-2/">CS231n notes on backprop</a></li>
        <li><a href="http://cs231n.stanford.edu/handouts/derivatives.pdf">Derivatives, Backpropagation, and Vectorization</a></li>
        <li><a href="http://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf">Learning Representations by Backpropagating Errors</a> (seminal Rumelhart et al. backpropagation paper)</li>
      </ol>
      Additional Readings:
      <ol>
        <li><a href="https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b">Yes you should understand backprop</a></li>
        <li><a href="http://www.jmlr.org/papers/volume12/collobert11a/collobert11a.pdf">Natural Language Processing (Almost) from Scratch</a></li>
      </ol>
    </td>
    <td>
      Assignment 2 <b><font color="green">out</font></b>
      <br>
      [<a href="assignments/a2.zip">code</a>]
      [<a href="assignments/a2.pdf">handout</a>]
    </td>
    <td>Assignment 1 <b><font color="red">due</font></b></td>
  </tr>

  <tr>
    <td>Dependency Parsing
      <br>
      [<a href="slides/cs224n-2021-lecture04-dep-parsing.pdf">slides</a>]
      [<a href="readings/cs224n-2019-notes04-dependencyparsing.pdf">notes</a>]
      <br>
      [<a href="slides/cs224n-2021-lecture04-dep-parsing-annotated.pdf">slides (annotated)</a>]
    </td>
    <td>
      Suggested Readings:
      <ol>
        <li><a href="https://www.aclweb.org/anthology/W/W04/W04-0308.pdf">Incrementality in Deterministic Dependency Parsing</a></li>
        <li><a href="https://www.emnlp2014.org/papers/pdf/EMNLP2014082.pdf">A Fast and Accurate Dependency Parser using Neural Networks</a></li>
        <li><a href="http://www.morganclaypool.com/doi/abs/10.2200/S00169ED1V01Y200901HLT002">Dependency Parsing</a></li>
        <li><a href="https://arxiv.org/pdf/1603.06042.pdf">Globally Normalized Transition-Based Neural Networks</a></li>
        <li><a href="http://nlp.stanford.edu/~manning/papers/USD_LREC14_UD_revision.pdf">Universal Stanford Dependencies: A cross-linguistic typology</a></li><a href="http://nlp.stanford.edu/~manning/papers/USD_LREC14_UD_revision.pdf">
        </a><li><a href="http://nlp.stanford.edu/~manning/papers/USD_LREC14_UD_revision.pdf"></a><a href="http://universaldependencies.org/">Universal Dependencies website</a></li>
      </ol>
    </td>
    <td></td>
    <td></td>
  </tr>

  <tr class="warning">
    <td>PyTorch Tutorial Session
    <br>
      [<a href="https://colab.research.google.com/drive/1Z6K6nwbb69XfuInMx7igAp-NNVj_2xc3?usp=sharing">colab notebook</a>]
      [<a href="materials/CS224N_PyTorch_Tutorial.html">preview</a>]
    <br>
      [<a href="materials/CS224N PyTorch Tutorial.ipynb">jupyter notebook</a>]
    </td>
    <td>
      <i class="fa fa-clock-o"></i> 10:00am - 11:20am
    </td>
    <td></td>
    <td></td>
  </tr>

  <tr>
    <td>Recurrent Neural Networks and Language Models
      <br>
      [<a href="slides/cs224n-2021-lecture05-rnnlm.pdf">slides</a>]
       [<a href="readings/cs224n-2019-notes05-LM_RNN.pdf">notes (lectures 5 and 6)</a>]
    </td>
    <td>
      Suggested Readings:
      <ol>
        <li><a href="https://web.stanford.edu/~jurafsky/slp3/3.pdf">N-gram Language Models</a> (textbook chapter)</li>
        <li><a href="http://karpathy.github.io/2015/05/21/rnn-effectiveness/">The Unreasonable Effectiveness of Recurrent Neural Networks</a> (blog post overview)</li>
        <!-- <li><a href="http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/">Recurrent Neural Networks Tutorial</a> (practical guide)</li> -->
        <li><a href="http://www.deeplearningbook.org/contents/rnn.html">Sequence Modeling: Recurrent and Recursive Neural Nets</a> (Sections 10.1 and 10.2)</li>
       <li><a href="http://norvig.com/chomsky.html">On Chomsky and the Two Cultures of Statistical Learning</a>
      </li></ol>
    </td>
    <td>Assignment 3 <b><font color="green">out</font></b>
        <br>
        [<a href="assignments/a3.zip">code</a>]
        [<a href="assignments/a3.pdf">handout</a>]
    </td>
    <td>Assignment 2 <b><font color="red">due</font></b></td>
  </tr>

  <tr>
    <td>Vanishing Gradients, Fancy RNNs, Seq2Seq
      <br>
      [<a href="slides/cs224n-2021-lecture06-fancy-rnn.pdf">slides</a>]
      [<a href="readings/cs224n-2019-notes05-LM_RNN.pdf">notes (lectures 5 and 6)</a>]
    </td>
    <td>
      Suggested Readings:
      <ol>
        <li><a href="http://www.deeplearningbook.org/contents/rnn.html">Sequence Modeling: Recurrent and Recursive Neural Nets</a> (Sections 10.3, 10.5, 10.7-10.12)</li>
        <li><a href="http://ai.dinfo.unifi.it/paolo//ps/tnn-94-gradient.pdf">Learning long-term dependencies with gradient descent is difficult</a> (one of the original vanishing gradient papers)</li>
        <li><a href="https://arxiv.org/pdf/1211.5063.pdf">On the difficulty of training Recurrent Neural Networks</a> (proof of vanishing gradient problem)</li>
        <li><a href="https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1174/lectures/vanishing_grad_example.html">Vanishing Gradients Jupyter Notebook</a> (demo for feedforward networks)</li>
        <li><a href="http://colah.github.io/posts/2015-08-Understanding-LSTMs/">Understanding LSTM Networks</a> (blog post overview)</li>
        <!-- <li><a href="https://arxiv.org/pdf/1504.00941.pdf">A simple way to initialize recurrent networks of rectified linear units</a></li> -->
      </ol>
    </td>
    <td></td>
    <td></td>
  </tr>

  <tr>
    <td>Machine Translation, Attention, Subword Models
      <br>
      [<a href="slides/cs224n-2021-lecture07-nmt.pdf">slides</a>]
      [<a href="readings/cs224n-2019-notes06-NMT_seq2seq_attention.pdf">notes</a>]
    </td>
    <td>
      Suggested Readings:
      <ol>
        <li><a href="https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1162/syllabus.shtml">Statistical Machine Translation slides, CS224n 2015</a> (lectures 2/3/4)</li>
        <li><a href="https://www.cambridge.org/core/books/statistical-machine-translation/94EADF9F680558E13BE759997553CDE5">Statistical Machine Translation</a> (book by Philipp Koehn)</li>
        <li><a href="https://www.aclweb.org/anthology/P02-1040.pdf">BLEU</a> (original paper)</li>
        <li><a href="https://arxiv.org/pdf/1409.3215.pdf">Sequence to Sequence Learning with Neural Networks</a> (original seq2seq NMT paper)</li>
        <li><a href="https://arxiv.org/pdf/1211.3711.pdf">Sequence Transduction with Recurrent Neural Networks</a> (early seq2seq speech recognition paper)</li>
        <li><a href="https://arxiv.org/pdf/1409.0473.pdf">Neural Machine Translation by Jointly Learning to Align and Translate</a> (original seq2seq+attention paper)</li>
        <li><a href="https://distill.pub/2016/augmented-rnns/">Attention and Augmented Recurrent Neural Networks</a> (blog post overview)</li>
        <li><a href="https://arxiv.org/pdf/1703.03906.pdf">Massive Exploration of Neural Machine Translation Architectures</a> (practical advice for hyperparameter choices)</li>
        <li><a href="https://arxiv.org/abs/1604.00788.pdf">Achieving Open Vocabulary Neural Machine Translation with Hybrid Word-Character Models</a></li>
        <li><a href="https://arxiv.org/pdf/1808.09943.pdf">Revisiting Character-Based Neural Machine Translation with Capacity and Compression</a></li>
      </ol>
    </td>
    <td>Assignment 4 <b><font color="green">out</font></b>
        <br>
        [<a href="assignments/a4.zip">code</a>]
        [<a href="assignments/a4.pdf">handout</a>]
        [<a href="https://docs.google.com/document/d/1BQOAjhBxWbywkB4rMFH9iinb6YHSjaWw1TOVlGfyYho/edit#heading=h.4tqnggp12z76">Azure Guide</a>]
        [<a href="https://docs.google.com/document/d/1jtANWXbIYXMZO_2X7jupauPxcEbz-TVJkdatg4gzOdk/edit">Practical Guide to VMs</a>]
    </td>
    <td>Assignment 3 <b><font color="red">due</font></b></td>
  </tr>

  <tr>
    <td>Final Projects: Custom and Default; Practical Tips
      <br>
      [<a href="slides/cs224n-2021-lecture08-final-project.pdf">slides</a>]
      [<a href="readings/final-project-practical-tips.pdf">notes</a>]
    </td>
    <td>
      Suggested Readings:
      <ol>
        <li><a href="https://www.deeplearningbook.org/contents/guidelines.html">Practical Methodology</a> (<i>Deep Learning</i> book chapter)</li>
      </ol>
    </td>
    <td>Project Proposal <b><font color="green">out</font></b>
        <br>
        [<a href="project/project-proposal-instructions-2021.pdf">instructions</a>]
        <br><br>
        Default Final Project <b><font color="green">out</font></b>
        <br>
        [<a href="project/default-final-project-handout-squad-track.pdf">handout (IID SQuAD track)</a>]
        <br>
        [<a href="project/default-final-project-handout-robustqa-track.pdf">handout (Robust QA track)</a>]
        <!--[<a href="https://github.com/minggg/squad">code</a>]-->
    </td>
    <td></td>
  </tr>

  <tr>
    <td>Transformers <i>(lecture by <a href="https://nlp.stanford.edu/~johnhew/">John Hewitt</a>)</i>
    <br>
    [<a href="slides/cs224n-2021-lecture09-transformers.pdf">slides</a>]
    [<a href="readings/cs224n-2019-notes07-QA.pdf">notes</a>]
    </td>
    <td>
      Suggested Readings:
      <ol>
        <li><a href="http://web.stanford.edu/class/cs224n/project/default-final-project-handout-squad-track.pdf">Project Handout (IID SQuAD track)</a>
        </li>
        <li><a href="http://web.stanford.edu/class/cs224n/project/default-final-project-handout-robustqa-track.pdf">Project Handout (Robust QA track)</a>
        </li>
        <li><a href="https://arxiv.org/abs/1706.03762.pdf">Attention Is All You Need</a>
        </li>
        <li><a href="https://jalammar.github.io/illustrated-transformer/">The Illustrated Transformer</a>
        </li>
        <li><a href="https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html">Transformer (Google AI blog post)</a>
        </li>
        <li><a href="https://arxiv.org/pdf/1607.06450.pdf">Layer Normalization</a></li>
        <li><a href="https://arxiv.org/pdf/1802.05751.pdf">Image Transformer</a></li>
        <li><a href="https://arxiv.org/pdf/1809.04281.pdf">Music Transformer: Generating music with long-term structure</a></li>
      </ol>
    </td>
    <td></td>
    <td></td>
  </tr>

  <tr>
    <td>More about Transformers and Pretraining <i>(lecture by <a href="https://nlp.stanford.edu/~johnhew/">John Hewitt</a>)</i>
      <br>
      [<a href="slides/cs224n-2021-lecture10-pretraining.pdf">slides</a>]
      [<a href="readings/cs224n-2019-notes07-QA.pdf">notes</a>]
    </td>
    <td>
    Suggested Readings:
      <ol>
        <li>
          <a href="https://arxiv.org/pdf/1810.04805.pdf">BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding</a>
        </li>
        <li>
           <a href="https://arxiv.org/abs/1902.06006.pdf">Contextual Word Representations: A Contextual Introduction</a>
        </li>
        <li><a href="http://jalammar.github.io/illustrated-bert/">The Illustrated BERT, ELMo, and co.</a></li>
      </ol>
    </td>
    <td>Assignment 5 <b><font color="green">out</font></b>
      <br>
      [<a href="assignments/a5.zip">code</a>]
      [<a href="assignments/a5.pdf">handout</a>]
    </td>
    <td>Assignment 4 <b><font color="red">due</font></b></td>
  </tr>

  <tr>
    <td>Question Answering <i>(guest lecture by <a href="https://www.cs.princeton.edu/~danqic/">Danqi Chen</a>)</i>
      <br>
      [<a href="slides/cs224n-2021-lecture11-qa-v2.pdf">slides</a>]
    </td>
    <td>
    Suggested Readings:
      <ol>
        <li>
          <a href="https://arxiv.org/pdf/1606.05250.pdf">SQuAD: 100,000+ Questions for Machine Comprehension of Text</a>
        </li>
        <li>
          <a href="https://arxiv.org/pdf/1611.01603.pdf">Bidirectional Attention Flow for Machine Comprehension</a>
        </li>
        <li>
          <a href="https://arxiv.org/pdf/1704.00051.pdf">Reading Wikipedia to Answer Open-Domain Questions</a>
        </li>
        <li>
          <a href="https://arxiv.org/pdf/1906.00300.pdf">Latent Retrieval for Weakly Supervised Open Domain Question Answering</a>
        </li>
        <li>
          <a href="https://arxiv.org/pdf/2004.04906.pdf">Dense Passage Retrieval for Open-Domain Question Answering</a>
        </li>
        <li>
          <a href="https://arxiv.org/pdf/2012.12624.pdf">Learning Dense Representations of Phrases at Scale</a>
        </li>
      </ol>
    </td>
    <td></td>
    <td>Project Proposal <b><font color="red">due</font></b></td>
  </tr>

  <tr>
    <td>Natural Language Generation <i>(lecture by <a href="https://atcbosselut.github.io/">Antoine Bosselut</a>)</i>
      <br>
      [<a href="slides/cs224n-2021-lecture12-generation.pdf">slides</a>]
    </td>
    <td>
    Suggested readings:
      <ol>
        <li>
        <a href="https://arxiv.org/abs/1904.09751.pdf">The Curious Case of Neural Text Degeneration</a>
        </li>
        <li>
        <a href="https://arxiv.org/abs/1704.04368.pdf">Get To The Point: Summarization with Pointer-Generator Networks</a>
        </li>
        <li>
        <a href="https://arxiv.org/abs/1805.04833.pdf">Hierarchical Neural Story Generation</a>
        </li>
        <li>
        <a href="https://arxiv.org/abs/1603.08023.pdf">How NOT To Evaluate Your Dialogue System</a>
        </li>
      </ol>
    </td>
    <td></td>
    <td></td>
  </tr>

  <tr>
    <td><i></i>
    </td>
    <td></td>
    <td>Project Milestone <b><font color="green">out</font></b>
    [<a href="project/project-milestone-instructions-2021.pdf">instructions</a>]
    </td>
    <td>Assignment 5 <b><font color="red">due</font></b></td>
  </tr>

  <tr>
      <td>Reference in Language and Coreference Resolution
        <br>
        [<a href="slides/cs224n-2021-lecture13-coref.pdf">slides</a>]
      </td>
    <td>
    Suggested readings:
      <ol>
        <li>
        <a href="https://web.stanford.edu/~jurafsky/slp3/22.pdf">Coreference Resolution chapter of Jurafsky and Martin</a>
        </li>
        <li>
          <a href="https://arxiv.org/pdf/1707.07045.pdf">End-to-end Neural Coreference Resolution</a>.
        </li>
      </ol>
    </td>
    <td></td>
    <td></td>
  </tr>

  <tr>
    <td>T5 and large language models: The good, the bad, and the ugly <i>(guest lecture by <a href="https://colinraffel.com/">Colin Raffel</a>)</i>
    <br>
    [<a href="slides/cs224n-2021-lecture14-t5.pdf">slides</a>]
    </td>
    <td>
    Suggested readings:
    <ol>
        <li>
        <a href="https://colinraffel.com/publications/jmlr2020exploring.pdf">Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer</a>
        </li>
    </ol>
    </td>
    <td></td>
    <td></td>
  </tr>

  <tr>
    <td>Integrating knowledge in language models <i>(lecture by <a href="http://www.mleszczy.com">Megan Leszczynski</a>)</i>
    <br>
    [<a href="slides/cs224n-2021-lecture15-lm.pdf">slides</a>]
    </td>
    <td>
    Suggested readings:
    <ol>
        <li>
        <a href="https://arxiv.org/pdf/1905.07129.pdf">ERNIE: Enhanced Language Representation with Informative Entities</a>
        </li>
        <li>
        <a href="https://arxiv.org/pdf/1906.07241.pdf">Barack’s Wife Hillary: Using Knowledge Graphs for Fact-Aware Language Modeling</a>
        </li>
        <li>
        <a href="https://arxiv.org/pdf/1912.09637.pdf">Pretrained Encyclopedia: Weakly Supervised Knowledge-Pretrained Language Model</a>
        </li>
        <li>
        <a href="https://www.aclweb.org/anthology/D19-1250.pdf">Language Models as Knowledge Bases?</a>
        </li>
    </ol>
    </td>
    <td></td>
    <td>Project Milestone <b><font color="red">due</font></b></td>
  </tr>

  <tr>
    <td>Social &amp; Ethical Considerations in NLP Systems <i>(guest lecture by <a href="http://www.cs.cmu.edu/~ytsvetko/">Yulia Tsvetkov</a>)</i>
    <br>
    [<a href="slides/cs224n-2021-lecture16-ethics.pdf">slides</a>]
    </td>
    <td></td>
    <td></td>
    <td></td>
  </tr>

  <tr>
    <td>Model Analysis and Explanation <i>(lecture by <a href="https://nlp.stanford.edu/~johnhew/">John Hewitt</a>)</i>
      <br>
      [<a href="slides/cs224n-2021-lecture17-analysis.pdf">slides</a>]
    </td>
    <td></td>
    <td></td>
    <td></td>
  </tr>

  <tr>
    <td>Future of NLP + Deep Learning <i>(lecture by <a href="https://murtyshikhar.github.io/">Shikhar Murty</a>)</i>
      <br>
      [<a href="slides/cs224n-2021-lecture18-future.pdf">slides</a>]
    </td>
    <td></td>
    <td></td>
    <td></td>
  </tr>

  <tr>
    <td></td>
    <td></td>
    <td>Project Summary Image and Paragraph <b><font color="green">out</font></b>
    [<a href="project/project-summary-instructions-2021.pdf">instructions</a>]
    </td>
    <td></td>
  </tr>

  <tr>
    <td>Ask Me Anything / Final Project Assistance</td>
    <td></td>
    <td></td>
    <td>
      Project <b><font color="red">due</font></b>
      [<a href="project/project-report-instructions-2021.pdf">instructions</a>]
    </td>
  </tr>

  <tr>
    <td>Final Project Emergency Assistance
    </td>
    <td></td>
    <td></td>
    <td></td>
  </tr>

  <tr>
    <td></td>
    <td></td>
    <td></td>
    <td>Project Summary Image and Paragraph <b><font color="red">due</font></b>
    </td>
  </tr>

  </tbody>
</table>
