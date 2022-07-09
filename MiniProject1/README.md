Hi, this is our repo for MiniProject1 about ResNet! The project is part of the course: ECE-GY 7123 Deep Learning 2022 Spring in NYU

Our final model was stored in the project1_model.py and the weights were stored in project1_model.pt (only containing weights of the model) and project1_final_model.pt (also include optimizer_state, accuracy, etc.).
Both files can be found under the root directory /MiniProject/

We have used normalization at the last step of data augmentation and the mean and std value were calculated as below,
transforms.Normalize((0.4244, 0.4146, 0.3836), (0.2539, 0.2491, 0.2420)).
We believe it's reasonable to do the exact same normalization on testdataset before testing.

The training procedure is in file project1_final_model.ipynb. Simply running through the whole ipynb file with trained weights project1_final_model.pt under the same folder will reproduce the same result as we described in the report. 
![alt text](https://github.com/Eziolin1/MiniProject1/blob/master/final_result.png)

If you want to retrain the model, you can just remove the project1_final_model.pt file and run through the training file again. But as we use SGD optimizer and initialize weight randomly, it's certain that every training process will lead to a differnent result and weights.
