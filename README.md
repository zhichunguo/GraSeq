# GraSeq: Graph and Sequence Fusion Learning for Molecular Property Prediction


## Introduction
This is the source code and dataset for the following paper: 

**GraSeq: Graph and Sequence Fusion Learning for Molecular Property Prediction. In CIKM 2020.**

Contact Zhichun Guo (zguo5@nd.edu), if you have any questions.

## Usage

### Installation
```
- torch >= 1.0.0
- sklearn >= 0.21.0
- rdkit.Chem 
```
### Run code

- For single-task classification (such as LogP, FDA, BBBP, BACE datasets):  
    ```
    python GraSeq_single/main.py
    ```
- For multi-task classification (such as Tox21 and ToxCast datasets):  
    ```
    python GraSeq_multi/main.py
    ```

## Reference

```
@inproceedings{guo2020graseq,
  title={GraSeq: Graph and Sequence Fusion Learning for Molecular Property Prediction},
  author={Guo, Zhichun and Yu, Wenhao and Zhang, Chuxu and Jiang, Meng and Chawla, Nitesh V},
  booktitle={Proceedings of the 29th ACM International Conference on Information \& Knowledge Management},
  pages={435--443},
  year={2020}
}
```