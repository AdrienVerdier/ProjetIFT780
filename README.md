# ProjetIFT780
Projet de Session IFT 780 - Réseaux Neuronaux

Realised by :

Alexandre TURPIN 
Quentin LEVIEUX 
Adrien VERDIER 

Here you will find the code of our project. In this file you can also find how to setup the development environment and how the project is structured.

## Setup
Create an isolated environment using your preferred solution (venv, pipenv,...) and install the required package:

```
pip install -r requirements.txt
```

## Run the project 

To run the project, you have to use the comand : 

```
python3 run.py 
```

Then, we add the parameters we want : 

- --model= : the model you want to use (AlexNet, VGGNet, ResNet, LeNet, ResNext)
- --dataset= : the dataset you want (cifar10, mnist)
- --optimizer= : Adam or SGD
- --validation_size= : the proportion of validation we want
- --batch_size= : the size of the batch
- --learning_rate= : the learning rate
- --num_epochs= : the number of epochs
- --metric= : the metric for active learning (entropy, margin, lc, random)
- --data_aug : if we want to use data augmentation
- --active_learning : if we want to activate active learning
- --generate_graphs1 : if we want to generate the graphs 1
- --generate_graphs2 : if we want to generate the graphs 2
- --pool_base= : the size of the pool base if we want to activate it
- --query_size= : the size of each query
- --size_data_first_learning= : the number of data we have at the beginning of the active learning
- --stop_criteria_step= : the number of step before we stop
- --stop_criteria_threshold= : the minimum evolution between the 2 last queries
- --stop_criteria_queries= : the size of the training set at the end of the active

For exemple, we can run this method : 

```
python3 run.py --model=LeNet --dataset=cifar10 --optimizer=Adam --batch_size=20 --num_epochs=10 --active_learning --metric=lc ----stop_criteria_step=15
```

This will launch the LeNet model on cifar10 with adam as an optimizer and a batch_size of 20. It will train 10 epochs on active learning with the least confident metric for 15 steps.

## Project structure

```
ProjetIFT780:
|   .gitignore
|   LICENSE
|   README.md
|   requirements.txt
|   run.ipynb
|   run.py
|   tree.txt
|   
+---data
|       .gitkeep
|                
+---graphs
|       .gitkeep
|       
\---src
    |   TrainManager.py
    |   utils.py
    |   __init__.py
    |   
    +---data
    |       DataManager.py
    |       
    +---features
    |       DataTransforms.py
    |       
    +---graphs
    |       GraphGenerator.py
    |       
    \---models
            AlexNet.py
            CNNBlocks.py
            LeNet.py
            ResNet.py
            ResNext.py
            VGG19.py
            VGGNet.py
            __init__.py
```