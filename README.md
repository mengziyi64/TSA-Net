# TSA-Net for CASSI
This repository contains the codes for paper **End-to-End Low Cost Compressive Spectral Imaging with Spatial-Spectral Self-Attention** (***ECCV (2020)***) by [Ziyi Meng*](https://github.com/mengziyi64), [Jiawei Ma*](https://github.com/Phoenix-V), [Xin Yuan](https://www.bell-labs.com/usr/x.yuan) (*Equal contributions). [[pdf]]()  
We provide simulation data and real data of our system. You can download them by the following link.  
[[Simu data (Google Drive)]](https://drive.google.com/drive/folders/1BNwkGHyVO-qByXj69aCf4SWfEsOB61J-?usp=sharing), [[Simu data (One Drive)]](https://1drv.ms/u/s!Au_cHqZBKiu2gYFDwE-7z1fzeWCRDA?e=ofvwrD), [[Simu data (Baidu Drive pw:aw5u)]](https://pan.baidu.com/s/1kWeH0IsHdj7Pbdd5oCJqTA)  
[[Real data (Google Drive)]](), [[Real data (One Drive)]](), [[Real data (Baidu Drive pw:)]]()


## Overviewer
Coded aperture snapshot spectral imaging ([CASSI](https://www.osapublishing.org/ao/abstract.cfm?uri=ao-47-10-B44)) is an effective tool to capture real-world 3D hyperspectral images. We have proposed a Spatial-Spectral Self-Attention module to jointly model the spatial and spectral correlation in an order-independent manner, which is incorporated in an encoder-decoder network to achieve high quality reconstruction for CASSI.

<p align="center">
<img src="Result/Images/setup.png" width="1200">
</p>
Fig. 1 (a) Single disperser coded aperture snapshot spectral imaging (SD-CASSI) and our experimental prototype. (b) 25 (out of 28) reconstructed spectral channels. (c) Principle of hardware coding.

## TSA-Net Architecture
<p align="center">
<img src="Result/Images/network.png" width="1200">
</p>
Fig. 2 (a) Spatial-Spectral Self-Attention (TSA) for one V feature (head). The spatial correlation involves the modelling for x-axis and y-axis separately and aggregation in an order-independent manner: the input is mapped to Q and K for each dimension: the size of kernel and feature are specified individually. The spectral correlation modelling will flatten samples in one spectral channel (2D plane) as a feature vector. The operation in dashed box denotes the network structure is shared while trained in parallel. (b)
TSA-Net Architecture. Each convolution layer adopts a 3 x 3 operator with stride 1 and outputs O-channel cube. The size of pooling and upsampling is P and T.
