# Joint-task Self-supervised Learning for Temporal Correspondence

[**Project**](https://sites.google.com/view/uvc2019) | [**Paper**]()

# Overview

[Joint-task Self-supervised Learning for Temporal Correspondence]()

[Xueting Li*](https://sunshineatnoon.github.io/), [Sifei Liu*](https://www.sifeiliu.net/), [Shalini De Mello](https://research.nvidia.com/person/shalini-gupta), [Xiaolong Wang](https://www.cs.cmu.edu/~xiaolonw/), [Jan Kautz](http://jankautz.com/), [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/).

(* equal contributions)

In  Neural Information Processing Systems (NeurIPS), 2019.

# Citation
If you use our code in your research, please use the following BibTex:

```
@inproceedings{uvc_2019,
    Author = {Xueting Li and Sifei Liu and Shalini De Mello and Xiaolong Wang and Jan Kautz and Ming-Hsuan Yang},
    Title = {Joint-task Self-supervised Learning for Temporal Correspondence},
    Booktitle = {NeurIPS},
    Year = {2019},
}
```
# Test on JHMDB
First propagate pose:
```
python test_jhmdb.py --evaluate --topk_vis 20 --sigma 0.5 --cropSize 320
```
Then test PCK by:
```
python eval_pck.py
```
Note you need to generate your own `testlist_split1.txt` w.r.t the JHMDB dataset path.

# Test on VIP
```
python test_VIP_instance_FAST_2stage.py --evaluate --cropSize 560 --cropSize2 80 --videoLen 2 --save_path VIP_ours --topk_vis 50 --predDistance 3 --save_path VIP_ours
```

and

```
python eval_instance2.py
```
Note you need to generate your own `vallist_ins.txt` w.r.t the VIP dataset path.

# Acknowledgements
- This code is based on [TPN](https://arxiv.org/pdf/1804.08758.pdf) and [TimeCycle](https://github.com/xiaolonw/TimeCycle).
- For any issues, please contact xli75@ucmerced.edu or sifeil@nvidia.com.
