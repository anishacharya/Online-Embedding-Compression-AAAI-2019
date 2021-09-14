[Online Embedding Compression for Text Classification Using Low Rank Matrix Factorization](https://ojs.aaai.org/index.php/AAAI/article/view/4578)
====================================
Acharya, Anish, Goel, Rahul, Metallinou, Angeliki, Dhillon, Inderjit


Abstract
===========
Deep learning models have become state of the art for natural language processing (NLP) tasks, however deploying these models in production system poses significant memory constraints. Existing compression methods are either lossy or introduce significant latency. We propose a compression method that leverages low rank matrix factorization during training, to compress the word embedding layer which represents the size bottleneck for most NLP models. Our models are trained, compressed and then further re-trained on the downstream task to recover accuracy while maintaining the reduced size. Empirically, we show that the proposed method can achieve 90% compression with minimal impact in accuracy for sentence classification tasks, and outperforms alternative methods like fixed-point quantization or offline word embedding compression. We also analyze the inference time and storage space for our method through FLOP calculations, showing that we can compress DNN models by a configurable ratio and regain accuracy loss without introducing additional latency compared to fixed point quantization. Finally, we introduce a novel learning rate schedule, the Cyclically Annealed Learning Rate (CALR), which we empirically demonstrate to outperform other popular adaptive learning rate algorithms on a sentence classification benchmark.

Media Coverage 
===============
[Packt](https://hub.packtpub.com/amazon-alexa-ai-researchers-develop-new-method-to-compress-neural-networks-and-preserves-accuracy-of-system/); 
[Alexa Blog](https://www.amazon.science/blog/new-method-for-compressing-neural-networks-better-preserves-accuracy)

Citation
===========
<pre>@inproceedings{acharya2019online,
  title={Online embedding compression for text classification using low rank matrix factorization},
  author={Acharya, Anish and Goel, Rahul and Metallinou, Angeliki and Dhillon, Inderjit},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  pages={6196--6203},
  year={2019}
}
</pre>


