# Leverage Lexical Knowledge for Chinese Named Entity Recognition via Collaborative Graph Network

Source code for "Leverage Lexical Knowledge for Chinese Named Entity Recognition via Collaborative Graph Network". If you use this code or our results in your research, we would appreciate it if you cite our paper as following:


```
@article{Sui2019Graph4CNER,
    title = {Leverage Lexical Knowledge for Chinese Named Entity Recognition via Collaborative Graph Network},
    author = {Sui, Dianbo and Chen, Yubo and Liu, Kang and Zhao, Jun and Liu, Shengping},
    journal = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing},
    year = {2019}
}
```
Requirement:
======
	Python: 3.7   
	PyTorch: 1.1.0 

Input format:
======
CoNLL format (We use BIO tag scheme), with each character its label for one line. Sentences are split with a null line.

	叶 B-PER
	嘉 I-PER
	莹 I-PER
	先 O
	生 O
	获 O
	聘 O
	南 B-ORG
	开 I-ORG
	大 I-ORG
	学 I-ORG
	终 O
	身 O
	校 O
	董 O
	。 O

Pretrained Embeddings:
====
Character embeddings (gigaword_chn.all.a2b.uni.ite50.vec): [Google Drive](https://drive.google.com/file/d/1_Zlf0OAZKVdydk7loUpkzD2KPEotUE8u/view?usp=sharing) or [Baidu Pan](https://pan.baidu.com/s/1pLO6T9D)

Word embeddings (sgns.merge.word): [Google Drive](https://drive.google.com/file/d/1Zh9ZCEu8_eSQ-qkYVQufQDNKPC4mtEKR/view) or
[Baidu Pan](https://pan.baidu.com/s/1luy-GlTdqqvJ3j-A4FcIOw)

How to run the code?
====
Details will be updated soon.



