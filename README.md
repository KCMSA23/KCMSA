# Knowledge-aware Prompt Learning Framework for Korean-Chinese Microblog Sentiment Analysis


## **:triangular_flag_on_post: This work has been accepted in ICASSP2024(IEEE International Conference on Acoustics, Speech and Signal Processing) [CCF-B]**
## **:triangular_flag_on_post: 🤗 Please [cite KCMSA](https://github.com/KCMSA23/KCMSA#-citing-kcmsa) in your publications if it helps with your work. It really means a lot to our open-source research. Thank you!**


## File Structure

* Code: Source code of our KAP implementation and baselines
* Data: Source files of our Korean-Chinese Microblog Sentiment Analysis (**KCMSA**) dataset and other dataset used in experiments
* Full version.pdf: 
    - Supplementary of Dataset Construction; 
    - Supplementary of Prompt Learning Framework; 
    - Supplementary of Experiments

## Demo Script Running
1. Running ourKAP framework over our KCMSA dataset
    ```
    python A_KAP.py --iter 0
    ```
2. Running our KAP framework over the KTEA dataset
    ```
    python A_KAP_twitter.py --iter 0
    ```

## Running the Scripts for All Five Seeds:
```
python exp_script.py
```

## Preliminary Steps
1. Download the '``config.json``', '``pytorch_model.bin``', '``special_tokens_map.json``', '``tokenizer_config.json``', '``tokenizer.json``', '``vocab.txt``' from https://huggingface.co/klue/bert-base/tree/main
2. Save the above five files in the "``Code/KLUE_BERT``" folder
3. Download the '``cc.ko.300.bin``' from https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ko.300.bin.gz and unzip the file
4. Save the file '``cc.ko.300.bin``' in the "``Code/Fasttext``" folder
5. (*For some baselines requiring the Hannanum tool*) Download the “KoNLPy” python package following https://konlpy.org/en/latest/install/
6. Create conda environment
    ```
    cd Code
    conda env create -f KCMSA_env.yml
    conda activate KCMSA_env
    ```
## ❖ Citing KCMSA
If you find KCMSA is helpful to your work, please cite our paper as below, 
⭐️star this repository, and recommend it to others who you think may need it. 🤗 Thank you!

```bibtex
@inproceedings{YANG24KCMSA,
  author       = {Xinyu Yang and
                  Hengxuan Wang and
                  Huiling Jin and
                  Zhenguo Zhang and
                  Xiaojie Yuan},
  title        = {Knowledge-aware Prompt Learning Framework for Korean-Chinese Microblog Sentiment Analysis},
  booktitle    = {{IEEE} International Conference on Acoustics, Speech and Signal Processing,
                  {ICASSP} 2024},
  publisher    = {{IEEE}},
  year         = {2024},
}
```

or

`Xinyu Yang, Hengxuan Wang, Huiling Jin, Zhenguo Zhang and Xiaojie Yuan. Knowledge-aware Prompt Learning Framework for Korean-Chinese Microblog Sentiment Analysis. 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2024.`


## Dataset Source
* KTEA: https://goo.gl/Gu0GNw

### Contact Us
yangxinyu@dbis.nankai.edu.cn
