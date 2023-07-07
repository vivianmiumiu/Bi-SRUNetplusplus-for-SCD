# Bi-SRUNetplusplus-for-SCD
Bi-SRUNetplusplus is a new method for multi-task change detection based on semantic change detection which achieves accuracy improvement by changing the backbone network.

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
