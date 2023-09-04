# Knowledge-aware Prompt Learning Framework for Korean-Chinese Microblog Sentiment Analysis

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
4. Save the above file in the "``Code/Fasttext``" folder
5. (*For some baselines requiring the Hannanum tool*) Download the “KoNLPy” python package following https://konlpy.org/en/latest/install/
6. Create conda environment
    ```
    cd Codes
    conda env create -f KCMSA_env.yml
    conda activate KCMSA_env
    ```

## Dataset Source
* KTEA: https://goo.gl/Gu0GNw

### Contact Us
yangxinyu@dbis.nankai.edu.cn
