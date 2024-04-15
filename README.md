# To execute 

## Install Libraries

pip install -r requirements.txt

## Train CBAGAN model

python train_sagan.py

## Test the model on a folder of images

python test_sagan.py

## Run IOU Evaluation Metric on a Folder of Images

python iou.py

## Run Dice Evaluation Metric on a Folder of Images

python dice.py

## Run FID Evaluation Metric on a Folder of Images

python fid.py

## Run RRT Algorithm on the Map

cd pathgan/models/rrt

python original_rrt.py

## Run CBAGAN-RRT Algorithm on the Map

cd ../../

python run_cbagan.py 

### An error will pop out on running the above code, please ignore it. The script is still saving the prediction mask correctly
 
cd models/rrt

python cbagan_rrt.py

### The results are stochastic on different runs because of the randomness in the RRT algorithm. In some cases the CBAGAN-RRT algorithm can take a longer path and more time than the original RRT algorithm.



