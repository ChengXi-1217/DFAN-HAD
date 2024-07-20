# Hyperspectral anomaly detection
This is the implementation of articles: ["Hyperspectral Anomaly Detection With Robust Graph Autoencoders"](https://ieeexplore.ieee.org/document/9494034) and ["Robust Graph Autoencoder for Hyperspectral Anomaly Detection"](https://ieeexplore.ieee.org/document/9414767).
# Usage
Run "**main.m**" after setting optimal parameters lambda, S and n_hid.
# Description
* **main.m** ---------- main file
  * **RGAE.m** ---------- implementation of the proposed algorithm;
    * **SuperGraph.m** ---------- construction of Laplacian matrix with SuperGraph;
      * **myPCA.m** ---------- PCA implementation;
    * **myRGAE.m** ---------- training of RGAE for hyperspectral anomaly detection;
  * **ROC.m**----------Calculate the AUC value with given detection map.
# Reference

If you find the code helpful, please kindly cite the following papers:
* Plain Text:<br>
  * G. Fan, Y. Ma, X. Mei, F. Fan, J. Huang and J. Ma, "Hyperspectral Anomaly Detection With Robust Graph Autoencoders," in IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2021.3097097
        
        
        
        .br>
  * G. Fan, Y. Ma, J. Huang, X. Mei and J. Ma, "Robust Graph Autoencoder for Hyperspectral Anomaly Detection," ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021, pp. 1830-1834, doi: 10.1109/ICASSP39728.2021.9414767
        
        
        
        .br>
* BibTeX:<br>
  * @ARTICLE{9494034,<br>
  author={G. {Fan} and Y. {Ma} and X. {Mei} and F. {Fan} and J. {Huang} and J. {Ma}},<br>
  journal={IEEE Transactions on Geoscience and Remote Sensing},<br>
  title={Hyperspectral Anomaly Detection With Robust Graph Autoencoders},<br>
  year={2021},<br>
  volume={},<br>
  number={},<br>
  pages={1-14}}<br>
  * @INPROCEEDINGS{9414767,<br>
  author={G. {Fan} and Y. {Ma} and J. {Huang} and X. {Mei} and J. {Ma}},<br>
  booktitle={ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},<br>
  title={Robust Graph Autoencoder for Hyperspectral Anomaly Detection},<br>
  year={2021},<br>
  volume={},<br>
  number={},<br>
  pages={1830-1834}}<br>
