# Sentimental

Sentimental is a project which allows to train a french sentimental analysis model based on CamemBert which is a french pre-trained transformer. The data used in this project is "Allociné" reviews so the model is well suited for movies sentiment analysis.

## 0- Clone the repository
```bash
git clone https://github.com/beroguedou/sentimental.git
```

## 1- Launch a training

Move in the cloned repository
```bash
cd sentimental
```

First of all you should create a directory with the name "saved" when the training function will save the future trained model. 
```bash
mkdir saved
```

And then you could run a training by running the following bash command:

```bash
python train.py --epochs 30 --batch 3 --workers 5 --stopping 5 --device 'cuda:0' --name "sentimental_camembert_model.pth"
```

## 2- Serve a prediction

To serve prediction we made a basic flask application that is connected to a model trained (model should be in the directory "saved").
```bash
python sentimental_french_app.py
```
