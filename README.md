# Bi-SRUNetplusplus-for-SCD
# Bi-SRNetUNet++ is based on the improvement of Bi-SRNetNet (Pytorch codes of 'Bi-Temporal Semantic Reasoning for the Semantic Change Detection in HR Remote Sensing Images' [[paper]](https://ieeexplore.ieee.org/document/9721305)


**Data preparation:**
1. Split the SCD data into training, validation and testing (if available) set and organize them as follows:

>YOUR_DATA_DIR
>  - Train
>    - im1
>    - im2
>    - label1
>    - label2
>  - Val
>    - im1
>    - im2
>    - label1
>    - label2
>  - Test
>    - im1
>    - im2
>    - label1
>    - label2
    
2. Find *-datasets -RS_ST.py*, set the data root in *Line 22* as *YOUR_DATA_DIR*
